using System.Collections.Generic;
using UnityEngine;

public class FaceTracker1D : MonoBehaviour
{
    public FaceDetectorProvider detector;
    public Transform hmd;
    public Camera centerEyeCam;
    public float detectHz = 12f;
    public float faceTimeoutSec = 0.6f;
    public float thetaSmoothAlpha = 0.2f;
    public float iouMatchThreshold = 0.25f;

    public readonly Dictionary<int, FaceTrack> faceDict = new Dictionary<int, FaceTrack>();

    float _nextDetectTime = 0f;
    int _nextId = 1;

    void Update()
    {
        if (!detector || !hmd || !centerEyeCam) return;

        if (Time.time < _nextDetectTime) { PruneStale(); return; }
        _nextDetectTime = Time.time + (1f / Mathf.Max(1f, detectHz));

        if (!detector.TryGetDetections(out var detections) || detections == null)
        {
            PruneStale();
            return;
        }

        var usedTracks = new HashSet<int>();

        foreach (var det in detections)
        {
            int bestId = -1;
            float bestIou = 0f;

            foreach (var kv in faceDict)
            {
                if (usedTracks.Contains(kv.Key)) continue;
                float iou = IoU(det.bbox01, kv.Value.bbox01);
                if (iou > bestIou)
                {
                    bestIou = iou;
                    bestId = kv.Key;
                }
            }

            if (bestId == -1 || bestIou < iouMatchThreshold)
            {
                bestId = _nextId++;
                faceDict[bestId] = new FaceTrack { trackId = bestId, thetaFaceDeg = 0f };
            }

            usedTracks.Add(bestId);

            var tr = faceDict[bestId];
            tr.bbox01 = det.bbox01;
            tr.lastSeenLocal = Time.time;

            Vector2 c = det.bbox01.center;
            Ray r = centerEyeCam.ViewportPointToRay(new Vector3(c.x, c.y, 0f));
            Vector3 dirLocal = hmd.InverseTransformDirection(r.direction).normalized;

            float thetaNew = AngleUtils.YawDegFromLocalDir_ZUp(dirLocal);
            tr.thetaFaceDeg = AngleUtils.SmoothAngleDeg(tr.thetaFaceDeg, thetaNew, thetaSmoothAlpha);

            faceDict[bestId] = tr;
        }

        PruneStale();
    }

    void PruneStale()
    {
        var toRemove = ListPool<int>.Get();
        foreach (var kv in faceDict)
        {
            if (Time.time - kv.Value.lastSeenLocal > faceTimeoutSec)
                toRemove.Add(kv.Key);
        }
        foreach (var id in toRemove) faceDict.Remove(id);
        ListPool<int>.Release(toRemove);
    }

    static float IoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.xMin, b.xMin);
        float y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);

        float interW = Mathf.Max(0f, x2 - x1);
        float interH = Mathf.Max(0f, y2 - y1);
        float inter = interW * interH;

        float union = a.width * a.height + b.width * b.height - inter;
        return union <= 1e-6f ? 0f : inter / union;
    }
}
