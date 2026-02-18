using System.Collections.Generic;
using UnityEngine;

// Output detections
public struct FaceDetection
{
    public Rect bbox;
    public float confidence;
}

public class FaceDetector : MonoBehaviour
{
    public bool TryGetDetections(out List<FaceDetection> detections)
    {
        // TODO: integrate face detector; return bbox list
        detections = null;
        return false;
    }
}
