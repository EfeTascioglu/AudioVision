using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class CaptionHUD : MonoBehaviour
{
    public CaptionStream client;
    public Transform hmd;

    public float anchorDistance = 3.0f;
    public float verticalOffset = -0.10f;

    public float horizontalFovDeg = 90f;
    public float yawSmoothingTau = 0.15f;

    public float staleSeconds = 0.8f;
    public float fadeOutSeconds = 0.8f;

    public Vector2 boxSize = new Vector2(150f, 30f);
    public float worldScale = 0.005f;
    public float fontSize = 18f;

    RectTransform _rootRect;
    Image _panelImg;
    TextMeshProUGUI _tmp;
    TextMeshProUGUI _arrowTmp;

    float _yawFiltered;
    float _lastMsgTime = -999f;
    string _latestText = "";

    Vector3 _dirQuest = Vector3.forward;
    bool _hasDirQuest;

    void Awake()
    {
        CreateWorldHud();
        ApplyAlpha(0f);
        SetArrow("");
    }

    void LateUpdate()
    {
        if (client == null || hmd == null) return;

        DrainIncoming();

        float age = Time.time - _lastMsgTime;
        float alpha = ComputeAlpha(age);
        ApplyAlpha(alpha);

        if (alpha <= 0.001f) return;
        if (!_hasDirQuest) return;

        Vector3 dirWorld = hmd.rotation * _dirQuest;

        transform.position = hmd.position + dirWorld * anchorDistance + hmd.up * verticalOffset;

        transform.rotation = Quaternion.LookRotation(transform.position - hmd.position, hmd.up);

        SetArrow(ComputeOffscreenArrow());
    }

    void DrainIncoming()
    {
        bool gotValidYaw = false;
        float yawCandidate = _yawFiltered;

        while (client.TryDequeueRawJson(out string json))
        {
            ServerPacket pkt;
            try
            {
                pkt = JsonUtility.FromJson<ServerPacket>(json);
            }
            catch
            {
                continue;
            }

            if (pkt != null && !string.IsNullOrWhiteSpace(pkt.transcription))
            {
                _latestText = pkt.transcription;
                if (_tmp != null) _tmp.text = _latestText;
                _lastMsgTime = Time.time;
            }

            if (pkt == null || pkt.localization == null || pkt.localization.Length < 3)
                continue;

            Vector3 mic = new Vector3(pkt.localization[0], pkt.localization[1], pkt.localization[2]);

            Vector3 quest = MicToQuest(mic);

            // Unity convention
            quest = new Vector3(quest.x, quest.z, quest.y);

            if (quest.sqrMagnitude < 1e-8f)
                continue;

            Vector3 dirQuest = quest.normalized;

            _dirQuest = dirQuest;
            _hasDirQuest = true;

            if (Mathf.Abs(dirQuest.x) > 1e-5f || Mathf.Abs(dirQuest.z) > 1e-5f)
            {
                float yaw = Mathf.Atan2(dirQuest.x, dirQuest.z);
                yawCandidate = yaw;
                gotValidYaw = true;
            }

            _lastMsgTime = Time.time;
        }

        if (!gotValidYaw) return;

        float dt = Mathf.Max(0f, Time.deltaTime);
        _yawFiltered = SmoothExp(_yawFiltered, yawCandidate, dt, yawSmoothingTau);
    }

    Vector3 MicToQuest(Vector3 vMic)
    {
        return vMic; // identity placeholder
    }

    string ComputeOffscreenArrow()
    {
        float halfFovRad = Mathf.Deg2Rad * horizontalFovDeg * 0.5f;
        if (Mathf.Abs(_yawFiltered) <= halfFovRad) return "";
        return (_yawFiltered > 0f) ? "→" : "←";
    }

    static float SmoothExp(float current, float target, float dt, float tau)
    {
        if (tau <= 1e-5f) return target;
        float a = 1f - Mathf.Exp(-dt / tau);
        return current + a * (target - current);
    }

    float ComputeAlpha(float age)
    {
        if (age <= staleSeconds) return 1f;
        float t = (age - staleSeconds) / Mathf.Max(1e-5f, fadeOutSeconds);
        return Mathf.Clamp01(1f - t);
    }

    void ApplyAlpha(float a)
    {
        if (_tmp != null)
        {
            var c = _tmp.color;
            c.a = a;
            _tmp.color = c;
        }

        if (_arrowTmp != null)
        {
            var c = _arrowTmp.color;
            c.a = a;
            _arrowTmp.color = c;
        }

        if (_panelImg != null)
        {
            var c = _panelImg.color;
            c.a = 0.55f * a;
            _panelImg.color = c;
        }
    }

    void SetArrow(string s)
    {
        if (_arrowTmp != null) _arrowTmp.text = s;
    }

    void CreateWorldHud()
    {
        gameObject.name = "CaptionHUD_World";
        int uiLayer = LayerMask.NameToLayer("UI");
        if (uiLayer != -1) gameObject.layer = uiLayer;

        var canvas = gameObject.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        _rootRect = gameObject.GetComponent<RectTransform>();
        if (_rootRect == null) _rootRect = gameObject.AddComponent<RectTransform>();
        _rootRect.sizeDelta = boxSize;

        var scaler = gameObject.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 1000f;

        var panelGO = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelGO.transform.SetParent(transform, false);
        if (uiLayer != -1) panelGO.layer = uiLayer;

        var panelRect = panelGO.GetComponent<RectTransform>();
        panelRect.sizeDelta = boxSize;

        _panelImg = panelGO.GetComponent<Image>();
        _panelImg.color = new Color(0f, 0f, 0f, 0.55f);

        var textGO = new GameObject("Text", typeof(RectTransform));
        textGO.transform.SetParent(panelGO.transform, false);
        if (uiLayer != -1) textGO.layer = uiLayer;

        _tmp = textGO.AddComponent<TextMeshProUGUI>();
        _tmp.text = "";
        _tmp.alignment = TextAlignmentOptions.Center;
        _tmp.fontSize = fontSize;
        _tmp.color = Color.white;
        _tmp.textWrappingMode = TextWrappingModes.NoWrap;

        var textRect = textGO.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(40f, 0f);
        textRect.offsetMax = new Vector2(-40f, 0f);

        var arrowGO = new GameObject("Arrow", typeof(RectTransform));
        arrowGO.transform.SetParent(panelGO.transform, false);
        if (uiLayer != -1) arrowGO.layer = uiLayer;

        _arrowTmp = arrowGO.AddComponent<TextMeshProUGUI>();
        _arrowTmp.text = "";
        _arrowTmp.alignment = TextAlignmentOptions.Center;
        _arrowTmp.fontSize = fontSize;
        _arrowTmp.color = Color.white;

        var arrowRect = arrowGO.GetComponent<RectTransform>();
        arrowRect.anchorMin = new Vector2(0.5f, 0.5f);
        arrowRect.anchorMax = new Vector2(0.5f, 0.5f);
        arrowRect.sizeDelta = new Vector2(80f, 80f);
        arrowRect.anchoredPosition = Vector2.zero;

        transform.localScale = Vector3.one * worldScale;
    }
}