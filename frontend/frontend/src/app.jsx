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

  return (
    <div className="dashboard">

      {/* Sidebar */}
      <div className="sidebar">
        <h2>🧪 BioLab</h2>
        <ul>
          <li className="active">Segmentation</li>
          <li>Analytics</li>
        </ul>
      </div>

      {/* Main */}
      <div className="main">
        <h1>Bacterial Growth Analysis</h1>

        {/* Top Section */}
        <form className="top-section" onSubmit={handleSubmit}>

         <div className="card">
  <h3>Upload Image</h3>

  <label className="upload-box">
    <input
      type="file"
      accept="image/*"
      onChange={handleFileChange}
      hidden
    />
    <span>📁 Upload Image</span>
  </label>

  {file && (
    <p className="file-name">✅ {file.name}</p>
  )}
</div>

          {/* Controls Card */}
          <div className="card">
            <h3>Controls</h3>

            <label>Resize Factor</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              value={resizeFactor}
              onChange={(e) => setResizeFactor(e.target.value)}
            />

            <label>Segmentation Method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {segmentationMethods.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>

            <label>Manual Threshold</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={manualThreshold}
              onChange={(e) => setManualThreshold(e.target.value)}
              disabled={
                method !== "Manual" &&
                method !== "Majority Fusion" &&
                method !== "CW-MTF (Novel)"
              }
            />

            <div className="checkbox">
              <input
                type="checkbox"
                checked={useWatershedSplit}
                onChange={(e) => setUseWatershedSplit(e.target.checked)}
              />
              <span>Watershed Split</span>
            </div>

            <button type="submit" disabled={loading}>
              {loading ? "Processing..." : "🚀 Analyze Image"}
            </button>
          </div>
        </form>

        {/* Error */}
        {error && <p className="error-text">{error}</p>}

        {/* Results */}
        {result && (
          <div className="results">

            <div className="card">
              <h3>Selected Method</h3>
              <p>{result.method || method}</p>
            </div>

            <div className="card">
              <h3>Metrics</h3>
              <p>Count: {result.metrics?.count ?? "N/A"}</p>
              <p>Total Area: {result.metrics?.total_area_px ?? "N/A"}</p>
              <p>Mean Area: {result.metrics?.mean_area_px ?? "N/A"}</p>
              <p>Median Area: {result.metrics?.median_area_px ?? "N/A"}</p>
              <p>Mean Circularity: {result.metrics?.mean_circularity ?? "N/A"}</p>
              <p>
                Mean Equivalent Diameter:{" "}
                {result.metrics?.mean_equiv_diameter_px ?? "N/A"}
              </p>
            </div>

            <div className="card">
              <h3>Fusion Weights</h3>
              {showWeights ? (
                <>
                  <p>Otsu: {result.weights?.otsu}</p>
                  <p>Adaptive: {result.weights?.adaptive}</p>
                  <p>Manual: {result.weights?.manual}</p>
                  <p>Color: {result.weights?.color}</p>
                </>
              ) : (
                <p>Only available for CW-MTF</p>
              )}
            </div>

            {/* Images */}
            <div className="image-grid">
              {[
                ["Resized", result.images?.resized],
                ["Grayscale", result.images?.grayscale],
                ["Preprocessed", result.images?.preprocessed],
                ["Color Likelihood", result.images?.color_likelihood],
                ["Binary Mask", result.images?.binary],
                ["Refined + Split", result.images?.refined_split],
                ["Segmented", result.images?.segmented],
              ].map(([title, path]) =>
                path ? (
                  <div className="image-card" key={title}>
                    <h4>{title}</h4>
                    <img src={makeImageUrl(path)} alt={title} />
                  </div>
                ) : null
              )}
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

export default App;