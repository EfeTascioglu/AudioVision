using UnityEngine;

public class FixedVectorHeadsetTest : MonoBehaviour
{
    public Transform hmd;                 // XR Camera
    public Vector3 fixedVectorHmd = new Vector3(0f, 0f, 1f); // forward in HMD-local
    public float distance = 2f;

    public float markerScale = 0.08f;
    private Transform marker;

    void Awake()
    {
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Destroy(sphere.GetComponent<Collider>());
        marker = sphere.transform;
        marker.localScale = Vector3.one * markerScale;
        marker.name = "TestMarker";
    }

    void Update()
    {
        if (!hmd) return;

        Vector3 dirWorld = hmd.TransformDirection(fixedVectorHmd.normalized);
        marker.position = hmd.position + dirWorld * distance;
    }
}
