using System;

namespace EGaze
{
    public interface ISource
    {
        bool Active { get; }

        string Name { get; }

        void Init();

        void Run();

        void Stop();

        void Pause();

        void Exit();

        void FlipVertical();

        void FlipHorizontal();

        event EventHandler OnStateChanged;
    }
}
