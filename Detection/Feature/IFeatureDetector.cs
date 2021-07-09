using System;
using System.Collections.Generic;

using EGaze.Source.Image;
using EGaze.OutputPublisher;

namespace EGaze.Detection.Feature
{
    public interface IFeatureDetector
    {
        IImageSource ImageSource { get; set; }

        event EventHandler OnDetectionFinished;
        IList<IFeature> DetectedFeatures { get; }
        IList<IImageSourceOutput> OutputPins { get; }
    }
}
