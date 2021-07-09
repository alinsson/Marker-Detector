
namespace EGaze.Source.Image
{
    public interface IImageSource : ISource
    {
        event ImageFrameReadyHandler OnImageFrame;

        ImageFrame CurrentFrame { get; }
    }
}
