using System.Collections.Generic;

public static class ListPool<T>
{
    static readonly Stack<List<T>> Pool = new Stack<List<T>>();

    public static List<T> Get() => Pool.Count > 0 ? Pool.Pop() : new List<T>();
    public static void Release(List<T> list) { list.Clear(); Pool.Push(list); }
}
