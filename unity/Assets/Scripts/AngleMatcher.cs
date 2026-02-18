using System.Collections.Generic;
using UnityEngine;

public class AngleMatcher1D : MonoBehaviour
{
    public SpeakerStore speakers;
    public FaceTracker1D faces;
    public float maxMatchDeg = 25f;
    public float toleranceDeg = 40f;
    public bool useSortedPairing = false;

    void Update()
    {
        if (!speakers || !faces) return;
        Match();
    }

    void Match()
    {
        var faceList = ListPool<FaceTrack>.Get();
        foreach (var kv in faces.faceDict) faceList.Add(kv.Value);

        var usedFaces = new HashSet<int>();

        foreach (var kv in speakers.speakerDict)
        {
            var sp = kv.Value;

            if (sp.matchedFaceId != -1 && faces.faceDict.TryGetValue(sp.matchedFaceId, out var curFace))
            {
                float err = AngleUtils.AngleErrorDeg(sp.thetaSoundDeg, curFace.thetaFaceDeg);
                if (err < toleranceDeg)
                {
                    usedFaces.Add(sp.matchedFaceId);
                    continue;
                }
            }

            int bestId = -1;
            float bestErr = 999f;

            for (int i = 0; i < faceList.Count; i++)
            {
                var f = faceList[i];
                if (usedFaces.Contains(f.trackId)) continue;

                float err = AngleUtils.AngleErrorDeg(sp.thetaSoundDeg, f.thetaFaceDeg);
                if (err < bestErr)
                {
                    bestErr = err;
                    bestId = f.trackId;
                }
            }

            if (bestId != -1 && bestErr <= maxMatchDeg)
            {
                sp.matchedFaceId = bestId;
                usedFaces.Add(bestId);
            }
            else
            {
                sp.matchedFaceId = -1;
            }
        }

        ListPool<FaceTrack>.Release(faceList);
    }
}
