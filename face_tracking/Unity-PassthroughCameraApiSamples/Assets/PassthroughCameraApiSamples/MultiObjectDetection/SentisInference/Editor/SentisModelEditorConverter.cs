// Copyright (c) Meta Platforms, Inc. and affiliates.
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection.Editor
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    [CustomEditor(typeof(SentisInferenceRunManager))]
    public class SentisModelEditorConverter : UnityEditor.Editor
    {
        // Output path for YOLOv9t-face model (1, 5, 8400): 4 box coords + 1 confidence score
        private const string FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/sentis_model.sentis";
        private SentisInferenceRunManager m_targetClass;
        private float m_iouThreshold;
        private float m_scoreThreshold;

        public void OnEnable()
        {
            m_targetClass = (SentisInferenceRunManager)target;
            m_iouThreshold = serializedObject.FindProperty("m_iouThreshold").floatValue;
            m_scoreThreshold = serializedObject.FindProperty("m_scoreThreshold").floatValue;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            if (GUILayout.Button("Generate YOLOv9/YOLOv9t-face Sentis model with Non-Max-Suppression layer"))
            {
                OnEnable(); // Get the latest values from the serialized object
                ConvertModel(); // convert the ONNX model to sentis
            }
        }

        private void ConvertModel()
        {
            //Load model
            var model = ModelLoader.Load(m_targetClass.OnnxModel);

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };
            var centersToCorners = Functional.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = Functional.Forward(model, input)[0];
            // YOLO output format: (1, 4+num_classes, N) channels-first
            // YOLOv8n COCO: (1, 84, 8400) = 4 box + 80 classes
            // YOLOv9t-face: (1, 5, 8400) = 4 box (cx,cy,w,h) + 1 for confidence score
            var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1); // cx, cy, w, h
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);
            var scores = Functional.ReduceMax(allScores, 1);                                    //shape=(N)
            var classIDs = Functional.ArgMax(allScores, 1);                                     //shape=(N)
            var corners = Functional.MatMul(boxCoords, centersToCorners);                    //shape=(N,4)
            var modelFinal = graph.Compile(corners, classIDs, scores);

            //Export the model to Sentis format
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref modelFinal);
            ModelWriter.Save(FILEPATH, modelFinal);

            // refresh assets
            AssetDatabase.Refresh();
        }
    }
}
