using System;
using System.Collections.Generic;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

using EGaze.Source.Image;
using EGaze.OutputPublisher;

namespace EGaze.Detection.Feature
{
    public class MarkerDetector : IFeatureDetector
    {

        #region Variables

        private bool _Debug, _FindSurface, _PerspectiveTransformation;
        private double _Epsilon;
        private int _MinArea, _CannyThreshold, _CannyThresholdLinking;
        private RetrType _RetrType;
        List<Marker> _Markers;
        Surface _Surface;
        int _ConfienceMax;
        Size _SurfaceSize;

        private OutputPin _CompositeOutput, _SurfaceOutput;
        private IImageSource _ImageSource;
        private IFeature[] _lDetectedFeatures;
        public IList<IImageSourceOutput> OutputPins { get; private set; }
        public event EventHandler OnDetectionFinished;

        #endregion

        #region Properties

        public bool Debug
        {
            get { return _Debug; }
            set { _Debug = value; }
        }

        public IImageSource ImageSource
        {
            get { return _ImageSource; }
            set { _ImageSource = value; }
        }

        public IList<IFeature> DetectedFeatures
        {
            get { return _lDetectedFeatures; }
        }

        #endregion

        public MarkerDetector(IImageSource imageSource, int MinArea, double Epsilon, int CannyThreshold, int CannyThresholdLinking, Size SurfaceSize, int ConfienceMax = 10, bool FindSurface = true, bool PerspectiveTransformation = true, bool soExternalRetangles = true)
        {
            _Debug = false;
            _Markers = new List<Marker>();
            _Surface = new Surface();

            this._ImageSource = imageSource;
            this._MinArea = MinArea;
            this._Epsilon = Epsilon;
            this._CannyThreshold = CannyThreshold;
            this._CannyThresholdLinking = CannyThresholdLinking;
            this._FindSurface = FindSurface;
            this._SurfaceSize = SurfaceSize;
            this._ConfienceMax = ConfienceMax;
            this._PerspectiveTransformation = PerspectiveTransformation;
            this._RetrType = soExternalRetangles ? RetrType.External : RetrType.List;

            _ImageSource.OnImageFrame += new ImageFrameReadyHandler(_ImageSource_OnImageFrame);

            _CompositeOutput = new OutputPin("Markers Detector", "Composite Markers");
            _SurfaceOutput = new OutputPin("Surface Detector", "Composite Surface");
            OutputPins = new List<IImageSourceOutput>();
            OutputPins.Add(_CompositeOutput);
            OutputPins.Add(_SurfaceOutput);
        }

        void _ImageSource_OnImageFrame(object sender, ImageFrameReadyEventArgs eventArgs)
        {
            var CompositeImage = eventArgs.ImageFrame.Mat.Clone();

            DetectionMarkers(CompositeImage);
            
            var countMarkers = _Markers.Count;
            if (countMarkers > 0)
            {
                _lDetectedFeatures = new IFeature[_Markers.Count + 1];

                if (_FindSurface && DetectionsSurface(CompositeImage, _PerspectiveTransformation))
                    _lDetectedFeatures[0] = _Surface;

                for (var i = 0; i < countMarkers; i++)
                {
                    PrintMarkerDirection(CompositeImage, _Markers[i], new Bgr(0, 255, 0), 2);
                    _lDetectedFeatures[(countMarkers > 2 && _FindSurface) ? i + 1 : i] = _Markers[i];
                }
            }

            // Trigger Event
            _CompositeOutput.UpdateImageFrame(CompositeImage);
            if (_Surface.Image != null)
                _SurfaceOutput.UpdateImageFrame(_Surface.Image);

            if (this.OnDetectionFinished != null)
                OnDetectionFinished(this, null);
        }

        public bool DetectionsSurface(Mat image, bool perspectiveTransformation = true)
        {
            _Surface.Detected = false;

            var countMarkers = _Markers.Count;
            if (countMarkers != 4)
                return true;
            else
            {
                #region Sample Surfice region

                //var centersMarkers = new Point[countMarkers];

                //for (var i = 0; i < countMarkers; i++)
                //    centersMarkers[i] = _Markers[i].Center;

                //var rectangle = CvInvoke.BoundingRectangle(new VectorOfPoint(centersMarkers));
                //CvInvoke.Rectangle(image, rectangle, new Bgr(127, 0, 0).MCvScalar, 2);
                //var rectangle = CvInvoke.MinAreaRect(new VectorOfPoint(centersMarkers));
                //var rectangle = CvInvoke.FitEllipse(new VectorOfPoint(centersMarkers));
                //rectangle.Angle += 90;

                //RotatedRectPrintVertices(image, rectangle.GetVertices(), new Bgr(255, 0, 0), 2);
                //var rectangle = CvInvoke.MinAreaRect(new VectorOfPoint(centersMarkers));
                //CvInvoke.Ellipse(image, rectangle, new MCvScalar(0, 0, 255), 2);
                //rectangle.Angle += 90;
                //RotatedRectPrintVertices(image, rectangle.GetVertices(), new Bgr(255, 0, 0), 2);
                //CvInvoke.Ellipse(image, rectangle, new MCvScalar(0, 255, 0), 2);
                //var a = rectangle.GetVertices();

                //return true;

                #endregion

                var centersMarkers = new Point[countMarkers];

                for (var i = 0; i < countMarkers; i++)
                    centersMarkers[i] = _Markers[i].Center;

                Point massCenter;
                PointF[] corners;
                SortCorners(centersMarkers, out corners, out massCenter);
                var perspective = GetPerspective(image, _SurfaceSize, corners);
                RotatedRectPrintVertices(image, corners, new Bgr(0, 127, 0), 2);

                _Surface.Image = perspective;

                return true;
            }

            //return false;
        }

        public void DetectionMarkers(Mat image, bool printMarkerFalse = true)
        {
            #region Convert to greyscale

            var grayImage = new Mat();
            if (image.NumberOfChannels == 1)
                grayImage = image;
            else if (image.NumberOfChannels == 3)
                CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            else
                throw new System.ArgumentException("Unsupported number of channels");

            #endregion

            #region Remove noise

            //UMat pyrDown = new UMat();
            //CvInvoke.PyrDown(grayImage, pyrDown);
            //CvInvoke.PyrUp(pyrDown, grayImage);
            //CvInvoke.Erode(grayImage, grayImage, null, new Point(-1, -1), 2, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);
            //CvInvoke.Dilate(grayImage, grayImage, null, new Point(-1, -1), 2, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);
            CvInvoke.GaussianBlur(grayImage, grayImage, new Size(), 1);

            #endregion

            #region Canny and Edge detection

            var cannyEdges = new Mat();
            CvInvoke.Canny(grayImage, cannyEdges, _CannyThreshold, _CannyThresholdLinking);
            var lines = CvInvoke.HoughLinesP(
               cannyEdges,
               1, //Distance resolution in pixel-related units
               Math.PI / 180.0, //Angle resolution measured in radians.
               20, //threshold
               30, //min Line width
               15); //gap between lines

            #endregion

            #region Find retangles

            var retangles = new List<Point[]>(); //4 points of retangle
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(cannyEdges, contours, null, _RetrType, ChainApproxMethod.ChainApproxSimple);
                var count = contours.Size;
                for (int i = 0; i < count; i++)
                {
                    using (var contour = contours[i])
                    using (var approxContour = new VectorOfPoint())
                    {
                        CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, false) * _Epsilon, true);
                        var a = CvInvoke.ContourArea(approxContour, false);
                        var tam = CvInvoke.ContourArea(approxContour, false);
                        if (tam > _MinArea) //only consider contours with area greater than 250
                            if (approxContour.Size == 4) //The contour has 4 vertices
                            {
                                var rectangle = CvInvoke.BoundingRectangle(approxContour);
                                if (!(rectangle.Height > 3 * rectangle.Width) || !(rectangle.Width > 3 * rectangle.Height))
                                    retangles.Add(approxContour.ToArray());
                            }
                    }
                }
            }

            #endregion

            #region Find Markers in all reactangles detecteds

            _Markers.Clear();
            foreach (var retangle in retangles)
            {
                // Get mass center
                Point massCenter;
                PointF[] corners;
                SortCorners(retangle, out corners, out massCenter);
                // Now the corners is perfectly sorted i.e. corners[0] = top-left, corners[1] = top-right, corners[2] = bottom-right, and corners[3] = bottom-left

                //Apply the perspective transformation
                var perspective = GetPerspective(image, new Size(50, 50), corners);
                CvInvoke.Resize(perspective, perspective, new Size(5, 5));

                // Decoder of image for find id of marker
                var marker = DecoderMarker(perspective, massCenter, corners);
                if (marker == null)
                {
                    if (printMarkerFalse) RotatedRectPrintVertices(image, retangle, new Bgr(0, 0, 127), 2);
                }
                else
                {
                    if (printMarkerFalse) RotatedRectPrintVertices(image, retangle, new Bgr(0, 127, 0), 2);
                    _Markers.Add(marker);
                }
            }

            #endregion
        }

        private Mat GetPerspective(Mat image, Size size, PointF[] corners)
        {
            var perspective = new Mat(size, Emgu.CV.CvEnum.DepthType.Cv8U, 1);

            // Corners of the destination image
            var quad = new PointF[4];
            quad[1] = new PointF(perspective.Cols, 0);
            quad[3] = new PointF(perspective.Cols, perspective.Rows);
            quad[2] = new PointF(0, perspective.Rows);

            // Get transformation matrix
            var transmtx = CvInvoke.GetPerspectiveTransform(corners, quad);

            // Apply perspective transformation
            CvInvoke.WarpPerspective(image, perspective, transmtx, perspective.Size);

            return perspective;
        }

        private void SortCorners(Point[] points, out PointF[] corners, out Point massCenter)
        {
            massCenter = new Point(); 
            // Get mass center
            foreach (var point in points)
            {
                massCenter.X += point.X;
                massCenter.Y += point.Y;
            }

            massCenter.X = (massCenter.X / points.Length);
            massCenter.Y = (massCenter.Y / points.Length);

            // Determine top-left, bottom-left, top-right, and bottom-right corner
            var bot = new Point[2];
            var top = new Point[2];
            var contTop = 0;
            var contBot = 0;
            foreach (var point in points)
            {
                if (point.Y < massCenter.Y)
                {
                    if (contTop == 2)
                        continue;
                    top[contTop++] = point;
                }
                else
                {
                    if (contBot == 2)
                        continue;
                    bot[contBot++] = point;
                }
            }

            corners = new PointF[4];
            corners[0] = top[0].X > top[1].X ? top[1] : top[0]; // tl
            corners[1] = top[0].X > top[1].X ? top[0] : top[1]; // tr
            corners[2] = bot[0].X > bot[1].X ? bot[1] : bot[0]; // bl
            corners[3] = bot[0].X > bot[1].X ? bot[0] : bot[1]; // br
        }

        public Marker DecoderMarker(Mat img, Point center, PointF[] corners)
        {
            var mId = 0;
            var grid = 5;
            int rotation = 0; // top = 0; right = 90; bottom = 180; left = 270
            var mWithB = new byte[grid, grid];
            var markerImg = img.ToImage<Gray, UInt16>();

            if (markerImg.Width != grid || markerImg.Height != grid) // image size error
            {
                if (_Debug) System.Diagnostics.Trace.WriteLine("Not found marker!");
                return null;
            }

            var msb = -1;
            for (var k = 0; k < 2; k++) // image rotated
            {
                for (var i = 0; i < grid; i++)
                {
                    for (var j = 0; j < grid; j++)
                    {
                        mWithB[i, j] = markerImg.Data[i, j, 0] < 100 ? (byte)0 : (byte)1;

                        if (i == 0 || j == 0 || i == grid - 1 || j == grid - 1)
                            if (mWithB[i, j] != (byte)0)
                            {
                                if (_Debug) System.Diagnostics.Trace.WriteLine("Not found marker!");
                                return null;
                            }
                    }
                }

                if ((mWithB[1, 1] == mWithB[3, 3]) && (mWithB[1, 1] == mWithB[1, 3]) && (mWithB[1, 1] != mWithB[3, 1]))
                {
                    msb = mWithB[1, 1];
                    break;
                }
                else // 3 possible image rotate
                {
                    if ((mWithB[1, 1] == mWithB[3, 1]) && (mWithB[1, 1] == mWithB[1, 3]) && (mWithB[1, 1] != mWithB[3, 3])) // 90
                    {
                        rotation = 360 - 90; // left <
                        markerImg = markerImg.Rotate(360 - rotation, new Gray(), false);
                    }
                    else if ((mWithB[1, 1] == mWithB[3, 3]) && (mWithB[1, 1] == mWithB[3, 1]) && (mWithB[1, 1] != mWithB[1, 3])) // 180
                    {
                        rotation = 360 - 180; // bottom v 
                        markerImg = markerImg.Rotate(360 - rotation, new Gray(), false);
                    }
                    else if ((mWithB[1, 3] == mWithB[3, 3]) && (mWithB[1, 3] == mWithB[3, 1]) && (mWithB[1, 1] != mWithB[3, 3])) // 270
                    {
                        rotation = 360 - 270; // right >
                        markerImg = markerImg.Rotate(360 - rotation, new Gray(), false);
                    }
                    else
                        return null;
                }
            }

            // m's
            var msg = new byte[grid];
            msg[4] = mWithB[1, 2];
            msg[3] = mWithB[2, 1];
            msg[2] = mWithB[2, 2];
            msg[1] = mWithB[2, 3];
            msg[0] = mWithB[3, 2];

            for (var i = 0; i < grid; i++)
                if (msg[i] == 1) // 0^0 = 1
                    mId += (int)Math.Pow(2, grid - 1 - i);

            if (msb == 0) // 0^0 = 1
                mId += (int)Math.Pow(2, grid);

            // Print diagnostics
            if (_Debug) System.Diagnostics.Trace.WriteLine("Decoder marker found " + mId + ":");

            // Now the corners is perfectly sorted i.e. corners[0] = top-left, corners[1] = top-right, corners[2] = bottom-right, and corners[3] = bottom-left
            var rect = new Rectangle(new Point((int)corners[0].X, (int)corners[0].Y), new Size((int)(corners[1].X - corners[0].X), (int)(corners[2].Y - corners[0].Y)));

            var marker = new Marker();
            marker.Id = mId;
            marker.Rotation = rotation;
            marker.Image = img;
            marker.Center = center;
            marker.Corners = corners;

            marker.Detected = true;
            marker.BoundingBox = rect;
            marker.Timestamp = DateTime.Now;

            return marker;
        }

        void RotatedRectPrintVertices(Mat img, Point[] vertices, Bgr bgr, int thickness = 1)
        {
            for (var i = 0; i < 3; i++)
                CvInvoke.Line(img, vertices[i], vertices[i + 1], bgr.MCvScalar, thickness);
            CvInvoke.Line(img, vertices[3], vertices[0], bgr.MCvScalar, thickness);
        }

        void RotatedRectPrintVertices(Mat img, PointF[] vertices, Bgr bgr, int thickness = 1)
        {
            // Determine top-left, top-right, bottom-left and bottom-right
            CvInvoke.Line(img, new Point((int)vertices[0].X, (int)vertices[0].Y), new Point((int)vertices[1].X, (int)vertices[1].Y), bgr.MCvScalar, thickness); // top
            CvInvoke.Line(img, new Point((int)vertices[1].X, (int)vertices[1].Y), new Point((int)vertices[3].X, (int)vertices[3].Y), bgr.MCvScalar, thickness); // bot
            CvInvoke.Line(img, new Point((int)vertices[0].X, (int)vertices[0].Y), new Point((int)vertices[2].X, (int)vertices[2].Y), bgr.MCvScalar, thickness); // left
            CvInvoke.Line(img, new Point((int)vertices[3].X, (int)vertices[3].Y), new Point((int)vertices[2].X, (int)vertices[2].Y), bgr.MCvScalar, thickness); // right
        }

        public void PrintMarkerDirection(Mat img, Marker marker, Bgr bgr, int thickness = 1)
        {
            RotatedRectPrintVertices(img, marker.Corners, bgr, thickness);

            if (marker.Rotation == 0) // top
            {
                var point = new Point(marker.Center.X, marker.Center.Y);
                CvInvoke.Line(img, new Point((int)marker.Corners[2].X, (int)marker.Corners[2].Y), new Point((int)point.X, (int)point.Y), bgr.MCvScalar, thickness); // bl - point
                CvInvoke.Line(img, new Point((int)point.X, (int)point.Y), new Point((int)marker.Corners[3].X, (int)marker.Corners[3].Y), bgr.MCvScalar, thickness); // point - br
            }
            else if (marker.Rotation == 90) // right
            {
                var point = new Point(marker.Center.X, marker.Center.Y);
                CvInvoke.Line(img, new Point((int)marker.Corners[0].X, (int)marker.Corners[0].Y), new Point((int)point.X, (int)point.Y), bgr.MCvScalar, thickness); // tl - point
                CvInvoke.Line(img, new Point((int)point.X, (int)point.Y), new Point((int)marker.Corners[2].X, (int)marker.Corners[2].Y), bgr.MCvScalar, thickness); // point - bl
            }
            else if (marker.Rotation == 180) // bottom
            {
                var point = new Point(marker.Center.X, marker.Center.Y);
                CvInvoke.Line(img, new Point((int)marker.Corners[1].X, (int)marker.Corners[1].Y), new Point((int)point.X, (int)point.Y), bgr.MCvScalar, thickness); // tr - point
                CvInvoke.Line(img, new Point((int)point.X, (int)point.Y), new Point((int)marker.Corners[0].X, (int)marker.Corners[0].Y), bgr.MCvScalar, thickness); // point - tl
            }
            else if (marker.Rotation == 270) // left
            {
                var point = new Point(marker.Center.X, marker.Center.Y);
                CvInvoke.Line(img, new Point((int)marker.Corners[3].X, (int)marker.Corners[3].Y), new Point((int)point.X, (int)point.Y), bgr.MCvScalar, thickness); // br - point
                CvInvoke.Line(img, new Point((int)point.X, (int)point.Y), new Point((int)marker.Corners[1].X, (int)marker.Corners[1].Y), bgr.MCvScalar, thickness); // point - tr
            }

            CvInvoke.PutText(img, (marker.Id).ToString(), marker.Center, FontFace.HersheyComplexSmall, 1, bgr.MCvScalar);
        }
    }
}