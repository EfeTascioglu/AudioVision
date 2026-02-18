using UnityEngine;

public static class AngleConversion
{
    // z-up convention: horizontal plane is x-y, angle is atan2(x, y)
    public static float CalculateAngle(Vector3 dirLocal)
    {
        dirLocal.z = 0f;
        
        // guard against division by zero
        if (dirLocal.sqrMagnitude < 1e-8f) return 0f;

        dirLocal.Normalize();
        return Mathf.Atan2(dirLocal.x, dirLocal.y) * Mathf.Rad2Deg;
    }

    public static float SmoothAngleDeg(float current, float target, float alpha)
    {
        return current + alpha * Mathf.DeltaAngle(current, target);
    }

    public static float AngleErrorDeg(float a, float b)
    {
        return Mathf.Abs(Mathf.DeltaAngle(a, b));
    }
}
