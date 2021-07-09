using System;
using EGaze.Source.Image;

namespace EGaze.OutputPublisher
{
    public interface IImageSourceOutput : IImageSource
    {
        string Description { get; }
    }
}
