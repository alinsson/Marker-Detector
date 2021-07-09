using System;
using System.Collections.Generic;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util;

using EGaze.Source.Image;
using EGaze.OutputPublisher;
using Emgu.CV.Util;
using System.Windows.Forms;

namespace EGaze.Detection.Feature
{
    public class ScreenDetector : IFeatureDetector
    {
        private const int MAXDetectedScreens = 1;

        private IFeature[] _lDetectedFeatures;
        private IImageSource _ImageSource;
        
        private OutputPin _CompositeOutput;
        private ImageFrame _CompositeOutputFrame;


        public IList<IImageSourceOutput> OutputPins { get; private set; }
        public event EventHandler OnDetectionFinished;

        #region Properties

        private Gray _CannyThreshold, _CannyThresholdLinking;
        private Rectangle _ScreenSizeMin, _ScreenSizeMax;
        private double _PerimeterMultiplier;

        public double PerimeterMultiplier
        {
            get { return _PerimeterMultiplier; }
            set { _PerimeterMultiplier = value; }
        }

        public Rectangle ScreenSizeMax
        {
            get { return _ScreenSizeMax; }
            set { _ScreenSizeMax = value; }
        }

        public Rectangle ScreenSizeMin
        {
            get { return _ScreenSizeMin; }
            set { _ScreenSizeMin = value; }
        }


        public double CannyThreshold
        {
          get { return _CannyThreshold.Intensity; }
          set { _CannyThreshold.Intensity = value; }
        }

        public double CannyThresholdLinking
        {
          get { return _CannyThresholdLinking.Intensity; }
          set { _CannyThresholdLinking.Intensity = value; }
        }

        public IList<IFeature> DetectedFeatures
        {
            get { return _lDetectedFeatures; }
        }

        public IImageSource ImageSource
        {
            get { return _ImageSource; }
            set { _ImageSource = value; }
        }

        #endregion

        public ScreenDetector(IImageSource imageSource)
        {
            this._ImageSource = imageSource;
            _lDetectedFeatures = new IFeature[MAXDetectedScreens];

            _CannyThreshold = new Gray(0);
            _CannyThresholdLinking = new Gray(0);

            _ScreenSizeMin = new Rectangle();
            _ScreenSizeMax = new Rectangle();

            PrepareDetectedObjects();

            if (_ImageSource != null)
                _ImageSource.OnImageFrame += new ImageFrameReadyHandler(_ImageSource_OnImageFrame);

            this._CompositeOutputFrame = new ImageFrame();

            _CompositeOutput = new OutputPin("Pupil Blob Detector Output Image", "Composite Pupil");

            OutputPins = new List<IImageSourceOutput>();
            OutputPins.Add(_CompositeOutput);
        }

        private void PrepareDetectedObjects()
        {
            // This method is used for optimization purpouses.
            // It could use a List<> variable size instaed, with higher memory and processing costs.
            // We expect that in the continous state there will be only one screen.
            for (int i = 0; i < MAXDetectedScreens; i++)
                _lDetectedFeatures[i] = new Screen();
        }

        bool _bFirstFrame = true;
        bool _bFirstOutputFrame = true;
        public event EventHandler OnFirstFrame;

        double _dScreenMinArea, _dScreenMaxArea;
        List<RotatedRect> _boxList = new List<RotatedRect>();

        void _ImageSource_OnImageFrame(object sender, ImageFrameReadyEventArgs eventArgs)
        {
            if (_bFirstFrame)
            {
                _SetupFirstFrame(eventArgs);
                _bFirstFrame = false;
            }

            if (eventArgs != null)
            {
                var CompositeImage = eventArgs.ImageFrame.Mat.Clone();

                var ProcImage = new Mat();
                CvInvoke.CvtColor(CompositeImage, ProcImage, ColorConversion.Bgr2Gray);

                var cannyEdges = new Mat();
                CvInvoke.PyrDown(ProcImage, cannyEdges);
                CvInvoke.Canny(cannyEdges, cannyEdges, _CannyThreshold.Intensity, _CannyThresholdLinking.Intensity); 
                CvInvoke.PyrUp(cannyEdges, cannyEdges);
                CvInvoke.Dilate(cannyEdges, cannyEdges, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

                // For search contours
                VectorOfPoint _biggestContour = null;

                using (var contours = new VectorOfVectorOfPoint())
                {
                    Mat aux = new Mat();
                    CvInvoke.FindContours(cannyEdges, contours, aux, Emgu.CV.CvEnum.RetrType.List, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

                    var _dArea = _dScreenMinArea;
                    var count = contours.Size;
                    for (var i = 0; i < count; i++)
                    {
                        using (var currentContour = contours[i])
                        {
                            var approxContour = new VectorOfPoint();
                            CvInvoke.ApproxPolyDP(currentContour, approxContour, CvInvoke.ArcLength(currentContour, true) * 0.05, true);
                            
                             // Only consider contours with area greater than _dArea and the contour has 4 vertices
                            var ContourArea = CvInvoke.ContourArea(approxContour, false);
                            if (ContourArea > _dArea && approxContour.Size == 4)
                            {
                                _dArea = ContourArea;
                                _biggestContour = approxContour;
                            }
                        }
                    }
                
                    if (_biggestContour != null)
                    {
                        var rect = CvInvoke.MinAreaRect(_biggestContour).MinAreaRect();
                        // Draw rentagle for 

                        for (int i = 0; i < 4; i++)
                            ((Screen)DetectedFeatures[0]).Corners[i] = _biggestContour[i];

                        ((Screen)_lDetectedFeatures[0]).Detected = true;
                        ((Screen)_lDetectedFeatures[0]).BoundingBox = rect;
                        ((Screen)_lDetectedFeatures[0]).Timestamp = DateTime.Now;

                        CvInvoke.Rectangle(CompositeImage, rect, new Bgr(Color.Orange).MCvScalar, 2);
                    }
                    else
                    {
                        ((Screen)DetectedFeatures[0]).Detected = false;
                    }
                }

                //Trigger Events
                _CompositeOutput.UpdateImageFrame(CompositeImage);

                if (_bFirstOutputFrame)
                {
                    _bFirstOutputFrame = false;
                    if (OnFirstFrame != null)
                        OnFirstFrame(this, null);
                }

                if (this.OnDetectionFinished != null)
                    OnDetectionFinished(this, null);
            }
        }

        private void _SetupFirstFrame(ImageFrameReadyEventArgs eventArgs)
        {
            _CannyThreshold.Intensity = 180;
            _CannyThresholdLinking.Intensity = 120;
            _PerimeterMultiplier = 0.08;

            _dScreenMinArea = ScreenSizeMin.Width * ScreenSizeMin.Height;
            _dScreenMaxArea = _ScreenSizeMax.Width * ScreenSizeMax.Height;
        }
    }
}
