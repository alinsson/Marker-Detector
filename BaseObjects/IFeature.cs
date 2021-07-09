using System;
using System.Drawing;

namespace EGaze
{
    public interface IFeature
    {
        bool Detected { get; set; }
        Rectangle BoundingBox { get; set; }
        FeatureType Type { get; set; }
        DateTime Timestamp { get; set; }

        //event EventHandler OnOffsetChange;
        //event EventHandler OnPositionChange;
        //event EventHandler OnTimestampChange;
    }
}
