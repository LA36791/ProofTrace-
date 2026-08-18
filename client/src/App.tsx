import { useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

type Evidence = {
  evidence_id: string;
  file_path: string;
  line_start?: number;
  line_end?: number;
  text: string;
  score?: number;
};

type Analysis = {
  query: string;
  outcome: "SUFFICIENT" | "INSUFFICIENT";
  reason: string;
  missing: string[];
  conclusion: {
    statement: string;
    evidence_ids: string[];
  } | null;
  llm_active?: boolean;
  llm_error?: string;
  retrieved_count?: number;
  selected_ids?: string[];
};

const EXAMPLE_QUERIES = [
  {
    label: "Trace discount flow",
    query:
      "Trace how a SAVE10 discount code flows from the cart through pricing and tax into the final order total.",
  },
  {
    label: "Investigate stock failure",
    query:
      "An order for an out-of-stock SKU-C item is failing. Which files show stock checking, reservation, and how the order placement reacts?",
  },
  {
    label: "Investigate discount incident",
    query:
      "Customers report a discount code SAVE20 is applied but the final charge is still the full price. What evidence explains why the discount is not reflected in the total?",
  },
  {
    label: "Understand payments",
    query:
      "How is a card charged and what happens when a charge is declined or invalid?",
  },
];

function App() {
  const [query, setQuery] = useState(EXAMPLE_QUERIES[1].query);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeExample, setActiveExample] = useState(1);

  async function investigate() {
    const trimmed = query.trim();

    if (!trimmed) {
      setError("Enter an investigation question first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const encoded = encodeURIComponent(trimmed);

      const [analysisResponse, retrievalResponse] = await Promise.all([
        fetch(
          `${API_BASE}/analyze?query=${encoded}&top_k=5&max_tokens=800`,
        ),
        fetch(
          `${API_BASE}/retrieve?query=${encoded}&top_k=5&mode=hybrid&max_tokens=800`,
        ),
      ]);

      if (!analysisResponse.ok) {
        throw new Error(
          `Analysis request failed with HTTP ${analysisResponse.status}`,
        );
      }

      if (!retrievalResponse.ok) {
        throw new Error(
          `Retrieval request failed with HTTP ${retrievalResponse.status}`,
        );
      }

      const analysisData = (await analysisResponse.json()) as Analysis;
      const retrievalData = await retrievalResponse.json();

      setAnalysis(analysisData);
      setEvidence(retrievalData.results ?? []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the ProofTrace backend.",
      );
      setAnalysis(null);
      setEvidence([]);
    } finally {
      setLoading(false);
    }
  }

  function chooseExample(index: number) {
    setActiveExample(index);
    setQuery(EXAMPLE_QUERIES[index].query);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">E</div>
          <div>
            <div className="brand-name">ProofTrace</div>
            <div className="brand-subtitle">AI evidence investigation</div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          SYSTEM ONLINE
          {analysis?.llm_active ? (
            <span className="mode-pill">AI GATE</span>
          ) : (
            <span className="mode-pill fallback">DETERMINISTIC</span>
          )}
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="eyebrow">EVIDENCE-FIRST DEBUGGING</div>
          <h1>Investigate with evidence, not assumptions.</h1>
          <p>
            Retrieve repository evidence, evaluate whether it is sufficient,
            and abstain when the available evidence cannot prove what happened.
          </p>
        </section>

        <section className="investigation-card">
          <div className="card-label">INVESTIGATION QUESTION</div>

          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Describe the software issue you want to investigate..."
            rows={4}
          />

          <div className="query-footer">
            <div className="examples">
              {EXAMPLE_QUERIES.map((example, index) => (
                <button
                  key={example.label}
                  className={`example-button ${
                    activeExample === index ? "active" : ""
                  }`}
                  onClick={() => chooseExample(index)}
                >
                  {example.label}
                </button>
              ))}
            </div>

            <button
              className="investigate-button"
              onClick={investigate}
              disabled={loading}
            >
              {loading ? "Investigating..." : "Investigate"}
              {!loading && <span>→</span>}
            </button>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <strong>Connection problem</strong>
            <span>{error}</span>
          </div>
        )}

        {!analysis && !loading && !error && (
          <section className="empty-state">
            <div className="empty-icon">⌕</div>
            <h2>Ready to investigate</h2>
            <p>
              Ask a question above or choose one of the example investigations
              to see the evidence gate in action.
            </p>
          </section>
        )}

        {loading && (
          <section className="loading-state">
            <div className="loader" />
            <div>
              <strong>Investigating repository evidence</strong>
              <p>
                Searching, ranking, budgeting context, and evaluating
                sufficiency...
              </p>
            </div>
          </section>
        )}

        {analysis && !loading && (
          <>
            <section className="result-grid">
              <div className="evidence-panel panel">
                <div className="panel-header">
                  <div>
                    <span className="panel-kicker">RETRIEVED EVIDENCE</span>
                    <h2>Evidence trace</h2>
                  </div>
                  <span className="count-badge">
                    {analysis.retrieved_count ?? evidence.length} retrieved
                  </span>
                </div>

                <div className="evidence-list">
                  {evidence.length === 0 ? (
                    <div className="no-evidence">
                      No repository evidence matched this investigation.
                    </div>
                  ) : (
                    evidence.map((item, index) => (
                      <EvidenceCard
                        key={item.evidence_id}
                        evidence={item}
                        index={index}
                        selected={analysis.selected_ids?.includes(
                          item.evidence_id,
                        )}
                      />
                    ))
                  )}
                </div>
              </div>

              <div className="gate-panel panel">
                <div className="panel-header">
                  <div>
                    <span className="panel-kicker">EVIDENCE GATE</span>
                    <h2>Decision</h2>
                  </div>
                  <span
                    className={`decision-badge ${
                      analysis.outcome === "SUFFICIENT"
                        ? "sufficient"
                        : "insufficient"
                    }`}
                  >
                    {analysis.outcome === "SUFFICIENT" ? "✓" : "!"}{" "}
                    {analysis.outcome}
                  </span>
                </div>

                <div className="decision-content">
                  <div className="decision-symbol">
                    {analysis.outcome === "SUFFICIENT" ? "✓" : "!"}
                  </div>

                  <h3>
                    {analysis.outcome === "SUFFICIENT"
                      ? "Evidence supports a conclusion"
                      : "The evidence is not sufficient"}
                  </h3>

                  <p className="decision-reason">{analysis.reason}</p>

                  {analysis.outcome === "INSUFFICIENT" &&
                    analysis.missing.length > 0 && (
                      <div className="missing-box">
                        <div className="missing-title">
                          Missing evidence
                        </div>

                        <ul>
                          {analysis.missing.map((item) => (
                            <li key={item}>
                              <span>+</span>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {analysis.conclusion && (
                    <div className="conclusion-box">
                      <div className="conclusion-title">
                        Evidence-backed conclusion
                      </div>
                      <p>{analysis.conclusion.statement}</p>

                      <div className="citation-row">
                        {analysis.conclusion.evidence_ids.map((id) => (
                          <span className="citation" key={id}>
                            {id}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>

            <section className="principle-banner">
              <div className="principle-icon">◈</div>
              <div>
                <strong>ProofTrace principle</strong>
                <span>
                  Relevant code explains what should happen. Runtime evidence
                  is required to prove what actually happened.
                </span>
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="footer">
        <span>ProofTrace</span>
        <span>Hybrid retrieval · Context budget · Evidence gate · Abstention</span>
      </footer>
    </div>
  );
}

function EvidenceCard({
  evidence,
  index,
  selected,
}: {
  evidence: Evidence;
  index: number;
  selected?: boolean;
}) {
  const filename = evidence.file_path.split(/[\\/]/).pop() ?? evidence.file_path;

  return (
    <article className={`evidence-card ${selected ? "selected" : ""}`}>
      <div className="evidence-number">
        {String(index + 1).padStart(2, "0")}
      </div>

      <div className="evidence-body">
        <div className="evidence-meta">
          <span className="file-name">{filename}</span>

          {evidence.line_start !== undefined && (
            <span className="line-range">
              lines {evidence.line_start}
              {evidence.line_end !== undefined &&
                `–${evidence.line_end}`}
            </span>
          )}

          {selected && <span className="selected-label">SELECTED</span>}
        </div>

        <div className="full-path">{evidence.file_path}</div>

        <pre>{evidence.text}</pre>

        <div className="evidence-id">{evidence.evidence_id}</div>
      </div>
    </article>
  );
}

export default App;
