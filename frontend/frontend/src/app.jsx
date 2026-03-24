import { useState } from "react";
import axios from "axios";
import "./app.css";

function App() {
  const [file, setFile] = useState(null);
  const [resizeFactor, setResizeFactor] = useState(1.0);
  const [manualThreshold, setManualThreshold] = useState(0.5);
  const [useWatershedSplit, setUseWatershedSplit] = useState(true);
  const [method, setMethod] = useState("CW-MTF (Novel)");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const segmentationMethods = [
    "Otsu",
    "Adaptive",
    "Manual",
    "Majority Fusion",
    "CW-MTF (Novel)",
  ];

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError("Please select an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("resize_factor", Number(resizeFactor));
    formData.append("manual_threshold", Number(manualThreshold));
    formData.append("use_watershed_split", useWatershedSplit);
    formData.append("method", method);

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const response = await axios.post(
        "http://127.0.0.1:8000/api/analyze/image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(
        err?.response?.data?.detail || "Failed to analyze image."
      );
    } finally {
      setLoading(false);
    }
  };

  const makeImageUrl = (path) => {
    if (!path) return null;
    return `http://127.0.0.1:8000/${path}`;
  };

  const showWeights = result?.weights;
  const showUncertainty = result?.images?.uncertainty;

  return (
    <div className="app-container">
      <h1>Bacterial Growth Segmentation</h1>

      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Select Image</label>
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </div>

        <div className="form-group">
          <label>Resize Factor</label>
          <input
            type="number"
            step="0.1"
            min="0.1"
            value={resizeFactor}
            onChange={(e) => setResizeFactor(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Segmentation Method</label>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {segmentationMethods.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Manual Threshold</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="1"
            value={manualThreshold}
            onChange={(e) => setManualThreshold(e.target.value)}
            disabled={method !== "Manual" && method !== "Majority Fusion" && method !== "CW-MTF (Novel)"}
          />
        </div>

        <div className="form-group checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={useWatershedSplit}
              onChange={(e) => setUseWatershedSplit(e.target.checked)}
            />
            Use Watershed Split
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Analyze Image"}
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}

      {result && (
        <div className="result-section">
          <h2>Analysis Result</h2>

          <div className="metrics-card">
            <h3>Selected Method</h3>
            <p><strong>Method:</strong> {result.method || method}</p>
          </div>

          <div className="metrics-card">
            <h3>Metrics</h3>
            <p><strong>Count:</strong> {result.metrics?.count ?? "N/A"}</p>
            <p><strong>Total Area:</strong> {result.metrics?.total_area_px ?? "N/A"}</p>
            <p><strong>Mean Area:</strong> {result.metrics?.mean_area_px ?? "N/A"}</p>
            <p><strong>Median Area:</strong> {result.metrics?.median_area_px ?? "N/A"}</p>
            <p><strong>Mean Circularity:</strong> {result.metrics?.mean_circularity ?? "N/A"}</p>
            <p>
              <strong>Mean Equivalent Diameter:</strong>{" "}
              {result.metrics?.mean_equiv_diameter_px ?? "N/A"}
            </p>
          </div>

          <div className="metrics-card">
            <h3>Fusion Weights</h3>
            {showWeights ? (
              <>
                <p><strong>Otsu:</strong> {result.weights?.otsu}</p>
                <p><strong>Adaptive:</strong> {result.weights?.adaptive}</p>
                <p><strong>Manual:</strong> {result.weights?.manual}</p>
                <p><strong>Color:</strong> {result.weights?.color}</p>
              </>
            ) : (
              <p>Fusion weights are available only for CW-MTF (Novel).</p>
            )}
          </div>

          <div className="image-grid">
            <div className="image-card">
              <h4>Resized</h4>
              <img src={makeImageUrl(result.images?.resized)} alt="Resized" />
            </div>

            <div className="image-card">
              <h4>Mask</h4>
              <img src={makeImageUrl(result.images?.mask)} alt="Mask" />
            </div>

            <div className="image-card">
              <h4>Uncertainty</h4>
              {showUncertainty ? (
                <img
                  src={makeImageUrl(result.images?.uncertainty)}
                  alt="Uncertainty"
                />
              ) : (
                <p>No uncertainty image for this method.</p>
              )}
            </div>

            <div className="image-card">
              <h4>Segmented</h4>
              <img
                src={makeImageUrl(result.images?.segmented)}
                alt="Segmented"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;