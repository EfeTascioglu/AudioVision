using System;
using System.Collections.Generic;
using UnityEngine;
public class SpeakerItem
{
    public string speaker_id;
    public float x;
    public float y;
    public string caption;
    public long seq;
}

public class SpeakerEnvelope
{
    public long server_ts_ms;
    public long seq;
    public SpeakerItem[] speakers;
}

public class SpeakerState
{
    public string speakerId;

    public float thetaSoundDeg;
    public float lastSeenLocal;
    public long lastServerTsMs;
    public long lastSeq;

    public int matchedFaceId = -1;

    public CaptionBuffer captions = new CaptionBuffer();
}

public class FaceTrack
{
    public int trackId;
    public Rect bbox;
    public float thetaFaceDeg;
    public float lastSeenLocal;
}
