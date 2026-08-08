"use client";
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from "recharts";

export default function Home() {
  const [activeTab, setActiveTab] = useState(0);

  // Tab 1 (Synthesis) State
  const [simLoading, setSimLoading] = useState(false);
  const [simData, setSimData] = useState(null);
  const [simConfig, setSimConfig] = useState({
    activity_type: "dynamic",
    sample_rate: 60,
    smoothing_n: 4,
    add_noise: true,
    duration_sec: 4.0
  });

  const handleSynthesize = async () => {
    setSimLoading(true);
    try {
      // In production this would point to your hosted FastAPI server (e.g. on Render)
      const res = await fetch("http://localhost:8000/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(simConfig)
      });
      const data = await res.json();
      setSimData(data);
    } catch (e) {
      console.error(e);
      alert("Failed to connect to backend API.");
    } finally {
      setSimLoading(false);
    }
  };

  // Convert raw 2D numpy arrays into Recharts format
  const formatChartData = (rawArray, dt) => {
    if (!rawArray) return [];
    return rawArray.map((row, i) => ({
      time: (i * dt).toFixed(2),
      x: row[0],
      y: row[1],
      z: row[2]
    }));
  };

  const renderSynthesisLab = () => {
    return (
      <div className="grid-2">
        <div className="panel">
          <h3 style={{marginBottom: "1.5rem"}}>🛠️ Simulation Controls</h3>
          
          <div className="form-group">
            <label>Motion Dynamic Style</label>
            <select style={{width: "100%"}} value={simConfig.activity_type} onChange={e => setSimConfig({...simConfig, activity_type: e.target.value})}>
              <option value="dynamic">dynamic (High Impact & Swings)</option>
              <option value="rhythmic">rhythmic (Running / Cycling)</option>
              <option value="complex">complex (Multi-stage Gym Workout)</option>
              <option value="static">static (Stationary / Stretching)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Sampling Frequency: {simConfig.sample_rate} Hz</label>
            <input type="range" min="40" max="150" step="10" value={simConfig.sample_rate} onChange={e => setSimConfig({...simConfig, sample_rate: parseInt(e.target.value)})} />
          </div>

          <div className="form-group">
            <label>Sequence Length: {simConfig.duration_sec}s</label>
            <input type="range" min="2" max="8" step="0.5" value={simConfig.duration_sec} onChange={e => setSimConfig({...simConfig, duration_sec: parseFloat(e.target.value)})} />
          </div>

          <div className="form-group" style={{display: "flex", gap: "10px", alignItems: "center", marginTop: "1.5rem"}}>
            <input type="checkbox" checked={simConfig.add_noise} onChange={e => setSimConfig({...simConfig, add_noise: e.target.checked})} id="noise" style={{width: "20px", height: "20px"}}/>
            <label htmlFor="noise" style={{margin: 0}}>Inject Physical PNP Noise</label>
          </div>

          <button className="btn-primary" style={{width: "100%", marginTop: "1.5rem"}} onClick={handleSynthesize} disabled={simLoading}>
            {simLoading ? "Synthesizing..." : "🔄 Synthesize Sequence"}
          </button>
        </div>

        <div className="panel">
          {simData ? (
            <div>
              <h3 style={{marginBottom: "1rem"}}>📐 Wrist Accelerometer (m/s²)</h3>
              <div style={{height: "300px", width: "100%"}}>
                <ResponsiveContainer>
                  <LineChart data={formatChartData(simData.wrist_acc, 1/simConfig.sample_rate)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)"/>
                    <XAxis dataKey="time" stroke="#cbd5e1" />
                    <YAxis stroke="#cbd5e1" />
                    <RechartsTooltip contentStyle={{backgroundColor: "#1E293B", border: "1px solid #38BDF8"}}/>
                    <Legend />
                    <Line type="monotone" dataKey="x" name="X-Axis" stroke="#38BDF8" dot={false} strokeWidth={2}/>
                    <Line type="monotone" dataKey="y" name="Y-Axis" stroke="#34D399" dot={false} strokeWidth={2}/>
                    <Line type="monotone" dataKey="z" name="Z-Axis" stroke="#F472B6" dot={false} strokeWidth={2}/>
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <h3 style={{marginBottom: "1rem", marginTop: "2rem"}}>🔄 Wrist Gyroscope (rad/s)</h3>
              <div style={{height: "300px", width: "100%"}}>
                <ResponsiveContainer>
                  <LineChart data={formatChartData(simData.wrist_gyro, 1/simConfig.sample_rate)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)"/>
                    <XAxis dataKey="time" stroke="#cbd5e1" />
                    <YAxis stroke="#cbd5e1" />
                    <RechartsTooltip contentStyle={{backgroundColor: "#1E293B", border: "1px solid #38BDF8"}}/>
                    <Legend />
                    <Line type="monotone" dataKey="x" name="Roll ω_x" stroke="#38BDF8" dot={false} strokeWidth={1.5}/>
                    <Line type="monotone" dataKey="y" name="Pitch ω_y" stroke="#34D399" dot={false} strokeWidth={1.5}/>
                    <Line type="monotone" dataKey="z" name="Yaw ω_z" stroke="#F472B6" dot={false} strokeWidth={1.5}/>
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div style={{display: "flex", height: "100%", alignItems: "center", justifyContent: "center", color: "var(--text-secondary)"}}>
              Click Synthesize to generate real-time 3D-to-IMU waveforms.
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <main className="container">
      <div className="title-container">
        <h1 className="main-title">SSSL-HAR: Synthetic-Data-Driven Self-Supervised Learning</h1>
        <div className="subtitle">
          ⚡ Recreating the IJCB 2025 Paper: Bridging Virtual Kinematic Synthesis & Multi-View Contrastive Learning (MVCL) for Flexible IMU Activity Recognition.
        </div>
      </div>

      <div className="grid-4">
        <div className="metric-card"><div className="card-title">Core Paradigm</div><div className="card-value">MVCL</div><div className="card-desc">Multi-View Contrastive Alignment</div></div>
        <div className="metric-card"><div className="card-title">Synthesis Engine</div><div className="card-value">3D-to-IMU</div><div className="card-desc">Smoothed Kinematic Simulation</div></div>
        <div className="metric-card"><div className="card-title">CroSSL-Synth</div><div className="card-value">88.15%</div><div className="card-desc">PAMAP2 Accuracy</div></div>
        <div className="metric-card"><div className="card-title">Data Efficiency</div><div className="card-value">~10 min</div><div className="card-desc">Minimal Finetuning</div></div>
      </div>

      <div className="tabs-container">
        {["💡 Why & What", "🔬 Synthesis Lab", "🚀 MVCL Studio", "📊 Benchmarks"].map((tab, i) => (
          <button key={i} className={`tab-btn ${activeTab === i ? "active" : ""}`} onClick={() => setActiveTab(i)}>
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 0 && (
        <div className="grid-2">
          <div className="panel" style={{borderColor: "rgba(248, 113, 113, 0.4)"}}>
            <h3 style={{color: "#F87171", marginBottom: "1rem"}}>🚨 The Real-World Bottleneck</h3>
            <ul style={{marginLeft: "1.5rem", lineHeight: "1.8"}}>
              <li><b>Data Collection is Exhausting:</b> Gathering labeled IMU sensor streams requires human subjects wearing rigid sensor rigs in controlled lab environments.</li>
              <li><b>Extreme Hardware Brittleness:</b> Standard neural networks fail completely when a user wears a different smartwatch brand, changes sampling rates, or shifts the strap half an inch.</li>
            </ul>
          </div>
          <div className="panel" style={{borderColor: "rgba(52, 211, 153, 0.4)"}}>
            <h3 style={{color: "#34D399", marginBottom: "1rem"}}>💡 The SSSL-HAR Breakthrough</h3>
            <ul style={{marginLeft: "1.5rem", lineHeight: "1.8"}}>
              <li><b>Zero-Cost Synthetic Data Pre-training:</b> We synthesize infinite inertial signals directly from publicly available 3D motion capture surface models.</li>
              <li><b>Universal Sensor Generalization:</b> By contrasting multiple synchronous sensor views, the AI learns biomechanically invariant motion profiles.</li>
            </ul>
          </div>
        </div>
      )}

      {activeTab === 1 && renderSynthesisLab()}
      
      {activeTab === 2 && (
        <div className="panel" style={{textAlign: "center", padding: "4rem"}}>
          <h3>🚀 Coming Soon</h3>
          <p style={{color: "var(--text-secondary)", marginTop: "1rem"}}>The MVCL t-SNE Studio is currently being migrated from Streamlit. Run the backend API to access the ML training routines.</p>
        </div>
      )}

      {activeTab === 3 && (
        <div className="panel">
          <h3 style={{marginBottom: "1.5rem"}}>📌 Table 1: Performance Comparison on PAMAP2 (3ACC+3GYRO)</h3>
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th>Acc</th>
                <th>Recall</th>
                <th>F1_M</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>SimCLR-real</td><td>75.69%</td><td>76.69%</td><td>76.03%</td></tr>
              <tr><td>SimCLR-synth</td><td>66.76%</td><td>59.46%</td><td>58.98%</td></tr>
              <tr><td>COCOA-real</td><td>87.07%</td><td>86.89%</td><td>87.37%</td></tr>
              <tr style={{background: "rgba(56, 189, 248, 0.15)"}}>
                <td style={{color: "#38BDF8", fontWeight: "bold"}}>CroSSL-synth (SSSL-HAR ⭐)</td>
                <td style={{color: "#38BDF8", fontWeight: "bold"}}>🔥 88.15%</td>
                <td style={{color: "#38BDF8", fontWeight: "bold"}}>87.39%</td>
                <td style={{color: "#38BDF8", fontWeight: "bold"}}>🔥 88.18%</td>
              </tr>
              <tr><td>Supervised baseline</td><td>92.46%</td><td>91.92%</td><td>92.02%</td></tr>
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
