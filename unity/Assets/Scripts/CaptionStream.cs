using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class CaptionStream : MonoBehaviour
{
    // Testing Server Messages
    public bool useTestData = true;
    public float testHz = 8f;

    [TextArea(2, 8)]
    public string[] testJsonMessages;

    // Queue for JSON strings
    private readonly ConcurrentQueue<string> _rawJsonQueue = new ConcurrentQueue<string>();
    private Coroutine _testRoutine;

    public string serverIP = "";
    public int serverPort = 0;
    public string serverPath = "/";

    private ClientWebSocket _ws;
    private CancellationTokenSource _wsCts;
    private Task _wsTask;

    void Start()
    {
        if (useTestData)
        {
            _testRoutine = StartCoroutine(TestFeedLoop());
        }
        else
        {
            // Real server connection placeholder.
            // Connect WebSocket and enqueue each received JSON string via EnqueueRawJson(json).
            string url = $"ws://{serverIP}:{serverPort}{serverPath}";
            _wsCts = new CancellationTokenSource();
            _wsTask = ConnectAndReceiveLoop(url, _wsCts.Token);
        }
    }

    void OnDisable()
    {
        if (_testRoutine != null) StopCoroutine(_testRoutine);
        _testRoutine = null;

        if (_wsCts != null)
        {
            _wsCts.Cancel();
            _wsCts.Dispose();
            _wsCts = null;
        }
    }

    public void EnqueueRawJson(string json)
    {
        if (!string.IsNullOrWhiteSpace(json))
            _rawJsonQueue.Enqueue(json);
    }

    public bool TryDequeueRawJson(out string json) => _rawJsonQueue.TryDequeue(out json);

    IEnumerator TestFeedLoop()
    {
        if (testJsonMessages == null || testJsonMessages.Length == 0)
        {
            // Default test stream: speaker moves from center -> right -> out-of-view right -> left -> out-of-view left
            testJsonMessages = new[]
            {
                "{\"localization\":[0.0,1.0,0.0],\"transcription\":\"Hi!\"}",
                "{\"localization\":[0.5,1.0,0.0],\"transcription\":\"Nice to meet you.\"}",
                "{\"localization\":[1.5,1.0,0.0],\"transcription\":\"My name is Bob.\"}",
                "{\"localization\":[3.0,0.7,0.0],\"transcription\":\"What is your name?\"}",
                "{\"localization\":[6.0,0.4,0.0],\"transcription\":\"That is a cool name.\"}",
                "{\"localization\":[-0.5,1.0,0.0],\"transcription\":\"Where are you from?\"}",
                "{\"localization\":[-1.5,1.0,0.0],\"transcription\":\"I am from Canada.\"}",
                "{\"localization\":[-3.0,0.7,0.0],\"transcription\":\"I study Engineering\"}",
                "{\"localization\":[-6.0,0.4,0.0],\"transcription\":\"at the University of Toronto.\"}"
            };
        }

        int i = 0;
        float period = 1f / Mathf.Max(0.1f, testHz);

        while (true)
        {
            EnqueueRawJson(testJsonMessages[i]);
            i = (i + 1) % testJsonMessages.Length;

            yield return new WaitForSeconds(period);
        }
    }

    private async Task ConnectAndReceiveLoop(string url, CancellationToken ct)
    {
    _ws = new ClientWebSocket();

    try
    {
        Debug.Log($"[CaptionStreamClient] Connecting to {url} ...");
        await _ws.ConnectAsync(new Uri(url), ct);
        Debug.Log("[CaptionStreamClient] WebSocket connected.");

        var buffer = new byte[64 * 1024];

        while (!ct.IsCancellationRequested && _ws.State == WebSocketState.Open)
        {
            var sb = new StringBuilder();
            WebSocketReceiveResult result;

            do
            {
                result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), ct);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    Debug.Log("[CaptionStreamClient] Server closed WebSocket.");
                    await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Server closed", CancellationToken.None);
                    return;
                }

                sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
            while (!result.EndOfMessage);

            string json = sb.ToString();
            if (!string.IsNullOrWhiteSpace(json))
            {
                EnqueueRawJson(json);
            }
        }
    }
    catch (OperationCanceledException)
    {
        Debug.Log("[CaptionStreamClient] WebSocket receive loop canceled.");
    }
    catch (Exception e)
    {
        Debug.LogWarning($"[CaptionStreamClient] WebSocket error: {e.Message}\n{e}");
    }
    finally
    {
        try
        {
            if (_ws != null)
            {
                if (_ws.State == WebSocketState.Open)
                    await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);

                _ws.Dispose();
                _ws = null;
            }
        }
        catch { }
    }
}
}