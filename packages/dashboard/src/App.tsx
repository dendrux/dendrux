import { Routes, Route, Navigate } from "react-router-dom";
import { RunsList } from "./pages/RunsList";
import { RunDetail } from "./pages/RunDetail";

export default function App() {
  return (
    <div className="min-h-screen bg-canvas">
      <Routes>
        <Route path="/" element={<RunsList />} />
        <Route path="/runs/:runId" element={<RunDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
