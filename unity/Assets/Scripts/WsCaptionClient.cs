using System.Collections.Concurrent;
using UnityEngine;

// Receives WebSocket messages from the web server and enqueues raw JSON files
public class WsCaptionClient : MonoBehaviour
{
    public string wsUrl = "";

    public readonly ConcurrentQueue<string> rawJsonQueue = new ConcurrentQueue<string>();

    void Start()
    {
        // TODO: Connect WebSocket here.
        // ws = new WebSocket(wsUrl);
        // ws.OnMessage += (msg) => rawJsonQueue.Enqueue(msg);
        // ws.Connect();
    }

    void OnDestroy()
    {
        // TODO: Close WS
    }
}
