using System;
using UnityEngine;

[Serializable]
public class ServerPacket
{
    // JSON output = {
    //     "localization": [x, y, z],
    //     "transcription": "..."
    // }
    public float[] localization;
    public string transcription;
}

public struct CaptionMessage
{
    public Vector3 localizationMic; // vector received from the microphone array
    public Vector3 localizationQuest; // vector after transformed to the Quest frame
    public float angle; // projected vector angle in the Quest frame
    public string text; // caption

    public static CaptionMessage FromPacket(ServerPacket pkt, Func<Vector3, Vector3> micToQuest)
    {
        Vector3 mic = Vector3.zero;
        if (pkt != null && pkt.localization != null && pkt.localization.Length >= 3)
        {   
            // check if the packet exists, the localization is not null, and the localization has at least three values
            mic = new Vector3(pkt.localization[0], pkt.localization[1], pkt.localization[2]);
        }

        // placeholder for transformation between microphone frame and Quest frame
        Vector3 quest = mic;

        float yaw = Mathf.Atan2(quest.x, quest.y);

        return new CaptionMessage
        {
            localizationMic = mic,
            localizationQuest = quest,
            angle = yaw,
            text = pkt != null ? (pkt.transcription ?? "") : "" // if pkt is null or transcription is null, text is ""
        };
    }
}