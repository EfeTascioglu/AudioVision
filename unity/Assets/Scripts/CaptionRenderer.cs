using System.Collections.Generic;
using UnityEngine;

public class CaptionRenderer : MonoBehaviour
{
    public SpeakerStore speakers;
    public FaceTracker1D faces;
    public Transform hmd;
    public Camera centerEyeCam;

    public CaptionLabelView labelPrefab;
    public Transform labelRoot;

    public float captionDisplaySeconds = 1.2f;

    private readonly Dictionary<string, CaptionLabelView> _views = new Dictionary<string, CaptionLabelView>();

    void Update()
    {
        if (!speakers || !faces || !hmd || !centerEyeCam || !labelPrefab) return;

        foreach (var kv in speakers.speakerDict)
        {
            if (!_views.ContainsKey(kv.Key))
            {
                var v = Instantiate(labelPrefab, labelRoot ? labelRoot : transform);
                v.name = $"Caption_{kv.Key}";
                _views[kv.Key] = v;
            }
        }

        var toRemove = ListPool<string>.Get();
        foreach (var kv in _views)
        {
            if (!speakers.speakerDict.ContainsKey(kv.Key))
                toRemove.Add(kv.Key);
        }
        foreach (var id in toRemove)
        {
            Destroy(_views[id].gameObject);
            _views.Remove(id);
        }
        ListPool<string>.Release(toRemove);

        foreach (var kv in speakers.speakerDict)
        {
            var id = kv.Key;
            var sp = kv.Value;

            sp.captions.Tick(Time.time, captionDisplaySeconds);

            var view = _views[id];
            view.SetText(sp.captions.Current);

            if (sp.matchedFaceId != -1 && faces.faceDict.TryGetValue(sp.matchedFaceId, out var ft))
            {
                Vector2 c = ft.bbox01.center;
                Ray r = centerEyeCam.ViewportPointToRay(new Vector3(c.x, c.y, 0f));
                Vector3 pos = r.origin + r.direction.normalized * 1.6f;
                Quaternion rot = Quaternion.LookRotation(pos - hmd.position, hmd.up);
                view.SetWorldPose(pos, rot);
            }
            else
            {
                float th = sp.thetaSoundDeg * Mathf.Deg2Rad;
                Vector3 dirLocal = new Vector3(Mathf.Sin(th), Mathf.Cos(th), 0f);
                Vector3 dirWorld = hmd.TransformDirection(dirLocal).normalized;

                Vector3 pos = hmd.position + dirWorld * 1.6f + hmd.up * 0.05f;
                Quaternion rot = Quaternion.LookRotation(pos - hmd.position, hmd.up);
                view.SetWorldPose(pos, rot);
            }
        }
    }
}
