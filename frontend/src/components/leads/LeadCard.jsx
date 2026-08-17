import { useState } from "react";
import Badge from "../ui/Badge.jsx";
import { TIER_LABELS, leadDisplayName } from "../../constants/labels.js";

export default function LeadCard({ lead, index = 0 }) {
  const [expanded, setExpanded] = useState(false);
  const name = leadDisplayName(lead);
  const tier = lead["AI Tier"];
  const score = lead["AI Score"];
  const rep = lead["Assigned Rep"];

  return (
    <article
      className="lead-card animate-slide-up"
      style={{ animationDelay: `${Math.min(index * 40, 400)}ms` }}
    >
      <div className="lead-card-top">
        <div className="lead-avatar">{name.charAt(0).toUpperCase()}</div>
        <div className="lead-card-main">
          <div className="lead-card-row">
            <h4 className="lead-name">{name}</h4>
            <div className="score-ring" data-tier={tier}>
              <span>{score ?? "—"}</span>
            </div>
          </div>
          <p className="lead-email">{lead.Email || "No email"}</p>
          <div className="lead-card-meta">
            <Badge tier={tier} />
            {rep && <span className="rep-chip">→ {rep}</span>}
          </div>
          {tier && TIER_LABELS[tier]?.hint && (
            <p className="lead-hint">{TIER_LABELS[tier].hint}</p>
          )}
          <p className="lead-action">{lead["Recommended Action"]}</p>
        </div>
      </div>
      {lead["Enrichment Summary"] && (
        <p className="lead-reasons">
          <strong>Pre-call brief</strong>
          {lead["Enrichment Confidence"] ? ` (${lead["Enrichment Confidence"]} confidence)` : ""}: {lead["Enrichment Summary"]}
        </p>
      )}
      {lead["AI Reasons"] && (
        <>
          <button type="button" className="link-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? "Hide details" : "Why this score?"}
          </button>
          {expanded && <p className="lead-reasons">{lead["AI Reasons"]}</p>}
        </>
      )}
    </article>
  );
}
