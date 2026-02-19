using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class FixedVectorHeadsetTest : MonoBehaviour
{
    public Transform hmd;
    public float distance = 1.5f;

    public string hudMessage = "Hello, world!";
    public float hudHeightOffset = 0.10f;

    public string worldMessage = "Very Cool!";
    public Vector3 worldPosition = new Vector3(0f, 1.3f, 2f);
    public bool worldBillboardToHmd = true;

    [Header("Style")]
    public Vector2 boxSize = new Vector2(420f, 120f);
    public float worldScale = 0.0015f;
    public float fontSize = 56f;

    private Transform hudLabel;
    private Transform worldLabel;

    void Awake()
    {
        hudLabel = CreateWorldLabel("HUD_Label", hudMessage);
        worldLabel = CreateWorldLabel("World_Label", worldMessage);

        worldLabel.position = worldPosition;
        worldLabel.rotation = Quaternion.identity;
    }

    void LateUpdate()
    {
        if (!hmd) return;

        if (hudLabel)
        {
            hudLabel.position = hmd.position + hmd.forward * distance + hmd.up * hudHeightOffset;
            hudLabel.rotation = Quaternion.LookRotation(hudLabel.position - hmd.position, hmd.up);
        }

        if (worldLabel)
        {
            worldLabel.position = worldPosition;

            if (worldBillboardToHmd)
                worldLabel.rotation = Quaternion.LookRotation(worldLabel.position - hmd.position, hmd.up);
        }
    }

    Transform CreateWorldLabel(string name, string text)
    {
        var root = new GameObject(name, typeof(RectTransform));
        int uiLayer = LayerMask.NameToLayer("UI");
        if (uiLayer != -1) root.layer = uiLayer;

        var canvas = root.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        var rootRect = root.GetComponent<RectTransform>();
        rootRect.sizeDelta = boxSize;

        var scaler = root.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 1000f;

        var panelGO = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelGO.transform.SetParent(root.transform, false);
        if (uiLayer != -1) panelGO.layer = uiLayer;

        var panelRect = panelGO.GetComponent<RectTransform>();
        panelRect.sizeDelta = boxSize;

        var panelImg = panelGO.GetComponent<Image>();
        panelImg.color = new Color(0f, 0f, 0f, 0.55f);

        var textGO = new GameObject("Text", typeof(RectTransform));
        textGO.transform.SetParent(panelGO.transform, false);
        if (uiLayer != -1) textGO.layer = uiLayer;

        var tmp = textGO.AddComponent<TextMeshProUGUI>();
        tmp.text = text;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.fontSize = fontSize;
        tmp.color = Color.white;
        tmp.textWrappingMode = TextWrappingModes.NoWrap;

        var textRect = textGO.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;

        root.transform.localScale = Vector3.one * worldScale;
        return root.transform;
    }
}
