using UnityEngine;
using TMPro;

public class CaptionLabelView : MonoBehaviour
{
    public TextMeshProUGUI tmp;

    public void SetText(string s)
    {
        if (!tmp) return;
        tmp.text = s;
        tmp.enabled = !string.IsNullOrEmpty(s);
    }
    public void SetViewportAnchor(Vector2 uv)
    {
        
    }
    public void SetWorldPose(Vector3 pos, Quaternion rot)
    {
        transform.position = pos;
        transform.rotation = rot;
    }
}
