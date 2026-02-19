using System.Collections.Generic;
using UnityEngine;

public class SpeakerStore : MonoBehaviour
{
    public WsCaptionClient ws;
    public Transform hmd;
    public Transform micFrame;

    [Header("Tuning")]
    public float thetaSmoothAlpha = 0.25f;
    public float speakerTimeoutSec = 2.0f;

    public readonly Dictionary<string, SpeakerState> speakerDict = new Dictionary<string, SpeakerState>();

    void Update()
    {
        if (!ws || !hmd || !micFrame) return;

        while (ws.rawJsonQueue.TryDequeue(out var json))
        {
            SpeakerEnvelope env;
            try { env = JsonUtility.FromJson<SpeakerEnvelope>(json); }
            catch { continue; }

            if (env?.speakers == null) continue;

            foreach (var s in env.speakers)
            {
                if (string.IsNullOrEmpty(s.speaker_id)) continue;

                long updateTs = env.server_ts_ms;
                long updateSeq = (s.seq != 0) ? s.seq : env.seq;
                bool hasSeq = updateSeq != 0;

                if (!speakerDict.TryGetValue(s.speaker_id, out var st))
                {
                    st = new SpeakerState { speakerId = s.speaker_id };
                    speakerDict[s.speaker_id] = st;
                }

                if (hasSeq)
                {
                    if (updateSeq <= st.lastSeq) continue;
                }
                else
                {
                    if (updateTs <= st.lastServerTsMs) continue;
                }

                if (hasSeq) st.lastSeq = updateSeq;
                st.lastServerTsMs = updateTs;
                st.lastSeenLocal = Time.time;

                Vector3 dirMic = new Vector3(s.x, s.y, 0f);
                if (dirMic.sqrMagnitude < 1e-8f) continue;
                dirMic.Normalize();

                Vector3 dirWorld = micFrame.TransformDirection(dirMic).normalized;
                Vector3 dirLocal = hmd.InverseTransformDirection(dirWorld).normalized;

                float thetaNew = AngleUtils.YawDegFromLocalDir_ZUp(dirLocal);
                st.thetaSoundDeg = AngleUtils.SmoothAngleDeg(st.thetaSoundDeg, thetaNew, thetaSmoothAlpha);

                if (!string.IsNullOrWhiteSpace(s.caption))
                    st.captions.Push(s.caption);
            }
        }

        PruneStale();
    }

    void PruneStale()
    {
        var toRemove = ListPool<string>.Get();
        foreach (var kv in speakerDict)
        {
            if (Time.time - kv.Value.lastSeenLocal > speakerTimeoutSec)
                toRemove.Add(kv.Key);
        }
        foreach (var id in toRemove) speakerDict.Remove(id);
        ListPool<string>.Release(toRemove);
    }
}
