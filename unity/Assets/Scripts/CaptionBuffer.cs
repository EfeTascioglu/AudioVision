using System.Collections.Generic;
using UnityEngine;

public class CaptionBuffer
{
    private readonly Queue<string> _q = new Queue<string>();
    private string _current = "";
    private float _until = 0f;
    private string _lastPushed = "";

    public string Current => _current;

    public void Push(string caption, int maxQueue = 12)
    {
        if (string.IsNullOrWhiteSpace(caption)) return;

        if (caption == _current) return;
        if (caption == _lastPushed) return;

        _q.Enqueue(caption);
        _lastPushed = caption;

        while (_q.Count > maxQueue) _q.Dequeue();
    }

    public void Tick(float now, float displaySeconds = 1.2f)
    {
        if (now < _until) return;

        if (_q.Count > 0)
        {
            _current = _q.Dequeue();
            _until = now + displaySeconds;
        }
        else
        {
            _current = "";
        }
    }
}
