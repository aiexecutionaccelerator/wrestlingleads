import { INBOX_RULE_ROWS, LEAD_SHARE_ROWS, leadShareTotal } from "../../constants/teamDistribution.js";

function NumInput({ id, value, onChange, min = 0, max = 100, step = 1, prefix, suffix }) {
  return (
    <div className="dist-num-wrap">
      {prefix ? (
        <span className="dist-affix" aria-hidden="true">
          {prefix}
        </span>
      ) : null}
      <input
        id={id}
        className="input dist-num-input"
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
      />
      {suffix ? (
        <span className="dist-affix" aria-hidden="true">
          {suffix}
        </span>
      ) : null}
    </div>
  );
}

export default function DistributionPanel({
  rules,
  rubric,
  coachingBoost,
  icpLlmMin,
  onRulesChange,
  onRubricChange,
  onCoachingBoostChange,
  onIcpLlmMinChange,
}) {
  const total = leadShareTotal(rules);
  const totalOk = Math.round(total) === 100;

  const ruleValue = (row) => {
    if (row.rulesKey) return rules?.[row.rulesKey] ?? "";
    if (row.rubricKey) return rubric[row.rubricKey] ?? "";
    if (row.scoringKey === "coaching_score_boost") return coachingBoost;
    if (row.scoringKey === "icp_llm_min") return icpLlmMin;
    return "";
  };

  const ruleChange = (row) => (e) => {
    const v = e.target.value;
    if (row.rulesKey) onRulesChange(row.rulesKey, v);
    else if (row.rubricKey) onRubricChange(row.rubricKey, v);
    else if (row.scoringKey === "coaching_score_boost") onCoachingBoostChange(v);
    else if (row.scoringKey === "icp_llm_min") onIcpLlmMinChange(v);
  };

  return (
    <div className="dist-panel">
      <section className="dist-section">
        <h3 className="dist-section-title">Who gets what % of new leads</h3>
        <p className="dist-section-desc">
          Compared against all scored leads in your inbox. Rows should add up to 100%.
        </p>
        <div className="dist-table-wrap">
          <table className="dist-table">
            <thead>
              <tr>
                <th>Who</th>
                <th>Band</th>
                <th className="dist-th-num">Share</th>
              </tr>
            </thead>
            <tbody>
              {LEAD_SHARE_ROWS.map((row) => (
                <tr key={row.rulesKey}>
                  <td className="dist-who">{row.who}</td>
                  <td className="dist-note">{row.note}</td>
                  <td className="dist-td-num">
                    <NumInput
                      id={`share-${row.rulesKey}`}
                      suffix="%"
                      min={0}
                      max={100}
                      value={rules?.[row.rulesKey] ?? ""}
                      onChange={(e) => onRulesChange(row.rulesKey, e.target.value)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <td colSpan={2} className="dist-total-label">
                  Total
                </td>
                <td className={`dist-total-value ${totalOk ? "ok" : "warn"}`}>
                  {total}% {totalOk ? "" : "— should be 100%"}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </section>

      <section className="dist-section">
        <h3 className="dist-section-title">Weekend split (Saturday &amp; Sunday, Eastern time)</h3>
        <p className="dist-section-desc">
          Optional. On weekends these replace Gene&apos;s and Jake&apos;s shares above — keep the two adding up to the
          same total as on weekdays so Beau &amp; Eric&apos;s band doesn&apos;t move. Leave blank to use the weekday shares.
        </p>
        <div className="dist-table-wrap">
          <table className="dist-table">
            <tbody>
              {[
                { key: "weekend_gene_pct", who: "Gene", note: "Top band on Sat/Sun" },
                { key: "weekend_jake_pct", who: "Jake", note: "Next band on Sat/Sun" },
              ].map((row) => (
                <tr key={row.key}>
                  <td className="dist-who">{row.who}</td>
                  <td className="dist-note">{row.note}</td>
                  <td className="dist-td-num">
                    <NumInput
                      id={`share-${row.key}`}
                      suffix="%"
                      min={0}
                      max={100}
                      value={rules?.[row.key] ?? ""}
                      onChange={(e) => onRulesChange(row.key, e.target.value)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="dist-section">
        <h3 className="dist-section-title">Inbox labels &amp; routing rules</h3>
        <p className="dist-section-desc">Same rules for everyone — how leads are labeled and when Gene jumps the line.</p>
        <div className="dist-table-wrap">
          <table className="dist-table">
            <thead>
              <tr>
                <th>Rule</th>
                <th>When it applies</th>
                <th className="dist-th-num">Value</th>
              </tr>
            </thead>
            <tbody>
              {INBOX_RULE_ROWS.map((row) => (
                <tr key={row.id}>
                  <td className="dist-who">{row.label}</td>
                  <td className="dist-note">{row.hint}</td>
                  <td className="dist-td-num">
                    <NumInput
                      id={`rule-${row.id}`}
                      prefix={row.prefix !== undefined ? row.prefix : "≥"}
                      min={row.min}
                      max={row.max}
                      value={ruleValue(row)}
                      onChange={ruleChange(row)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
