using System;

namespace EGaze.Source.Image
{
    public class ImageFrame
    {
        public int FrameNumber { get; set; }
        public DateTime Timestamp { get; set; }
        public Emgu.CV.Mat Mat {get; set; }
        public System.Drawing.Bitmap Bitmap { get { return this.Mat.Bitmap; } }
        public int Height { get { return this.Mat.Height; } }
        public int Width { get  { return this.Mat.Width; } }
    }
}
