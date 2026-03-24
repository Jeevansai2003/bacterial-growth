import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [resizeFactor, setResizeFactor] = useState(1.0);
  const [manualThreshold, setManualThreshold] = useState(0.5);
  const [useWatershedSplit, setUseWatershedSplit] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

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
    formData.append("resize_factor", resizeFactor);
    formData.append("manual_threshold", manualThreshold);
    formData.append("use_watershed_split", useWatershedSplit);

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
      setError("Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  const makeImageUrl = (path) => {
    if (!path) return null;
    return `http://127.0.0.1:8000/${path}`;
  };

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
            value={resizeFactor}
            onChange={(e) => setResizeFactor(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Manual Threshold</label>
          <input
            type="number"
            step="0.1"
            value={manualThreshold}
            onChange={(e) => setManualThreshold(e.target.value)}
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
            <h3>Metrics</h3>
            <p><strong>Count:</strong> {result.metrics?.count}</p>
            <p><strong>Total Area:</strong> {result.metrics?.total_area_px}</p>
            <p><strong>Mean Area:</strong> {result.metrics?.mean_area_px}</p>
            <p><strong>Median Area:</strong> {result.metrics?.median_area_px}</p>
            <p><strong>Mean Circularity:</strong> {result.metrics?.mean_circularity}</p>
            <p><strong>Mean Equivalent Diameter:</strong> {result.metrics?.mean_equiv_diameter_px}</p>
          </div>

          <div className="metrics-card">
            <h3>Fusion Weights</h3>
            <p><strong>Otsu:</strong> {result.weights?.otsu}</p>
            <p><strong>Adaptive:</strong> {result.weights?.adaptive}</p>
            <p><strong>Manual:</strong> {result.weights?.manual}</p>
            <p><strong>Color:</strong> {result.weights?.color}</p>
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
              {result.images?.uncertainty ? (
                <img
                  src={makeImageUrl(result.images?.uncertainty)}
                  alt="Uncertainty"
                />
              ) : (
                <p>No uncertainty image</p>
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