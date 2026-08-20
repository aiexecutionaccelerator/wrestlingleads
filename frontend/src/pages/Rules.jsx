import { useCallback, useEffect, useState } from "react";
import {
  fetchHubspotOwners,
  fetchJson,
  fetchRoutingRules,
  fetchRoutingStats,
  fetchScoringRubric,
  fetchWufooForms,
  saveRoutingRules,
  saveScoringRubric,
  sendRoute,
  sendUnrouted,
} from "../api.js";
import Card from "../components/ui/Card.jsx";
import Toggle from "../components/ui/Toggle.jsx";
import Toast from "../components/ui/Toast.jsx";
import LoadingSkeleton from "../components/ui/LoadingSkeleton.jsx";
import Badge from "../components/ui/Badge.jsx";
import { BUCKET_INFO } from "../constants/labels.js";
import DistributionPanel from "../components/team/DistributionPanel.jsx";

function repBucketClass(bucket) {
  return BUCKET_INFO[bucket]?.color || "rep-general";
}

function parseWeeklyCap(value) {
  if (value === "" || value == null) return null;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return null;
  return Math.floor(n);
}

function parseWeight(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return 1;
  return n;
}

function sharePercent(rep, reps) {
  const pool = (reps || []).filter((r) => r.bucket === rep.bucket && r.bucket !== "automation");
  const total = pool.reduce((sum, r) => sum + parseWeight(r.weight ?? 1), 0);
  if (!total || pool.length < 2) return null;
  return Math.round((parseWeight(rep.weight ?? 1) / total) * 100);
}

export default function Rules() {
  const [rules, setRules] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [testEmail, setTestEmail] = useState("");
  const [rubric, setRubric] = useState({ Hot: 68, Warm: 42, Cold: 18 });
  const [coachingBoost, setCoachingBoost] = useState(8);
  const [icpLlmMin, setIcpLlmMin] = useState(68);
  const [hubspotOwners, setHubspotOwners] = useState([]);
  const [wufooForms, setWufooForms] = useState([]);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [r, s, rubricData, formsData] = await Promise.all([
        fetchRoutingRules(),
        fetchRoutingStats(),
        fetchScoringRubric().catch(() => null),
        fetchWufooForms().catch(() => ({ forms: [] })),
      ]);
      setRules(r);
      setStats(s);
      setWufooForms(formsData?.forms || []);
      if (rubricData?.tiers) setRubric(rubricData.tiers);
      if (rubricData?.coaching_score_boost != null) setCoachingBoost(rubricData.coaching_score_boost);
      if (rubricData?.icp_llm_min != null) setIcpLlmMin(rubricData.icp_llm_min);
    } catch (err) {
      setError(err.message || "Could not load team settings");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (rules?.hubspot_configured) {
      fetchHubspotOwners()
        .then((data) => setHubspotOwners(data?.owners || []))
        .catch(() => setHubspotOwners([]));
    }
  }, [rules?.hubspot_configured]);

  const updateRep = (index, field, value) => {
    setRules((prev) => {
      const reps = [...(prev.reps || [])];
      reps[index] = { ...reps[index], [field]: value };
      return { ...prev, reps };
    });
  };

  const handleSave = async () => {
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const payload = {
        ...rules,
        reps: (rules.reps || []).map((rep) => ({
          ...rep,
          weekly_cap: parseWeeklyCap(rep.weekly_cap),
          weight: parseWeight(rep.weight ?? 1),
        })),
      };
      const saved = await saveRoutingRules(payload);
      const rubricResult = await saveScoringRubric({
        tiers: {
          Hot: Number(rubric.Hot),
          Warm: Number(rubric.Warm),
          Cold: Number(rubric.Cold),
        },
        coaching_score_boost: Number(coachingBoost),
        icp_llm_min: Number(icpLlmMin),
      });
      setRules(saved);
      if (rubricResult?.tiers) setRubric(rubricResult.tiers);
      const relabeled = rubricResult?.leads_relabeled ?? 0;
      setMessage(
        relabeled > 0
          ? `Saved — ${relabeled.toLocaleString()} leads updated with new priority labels.`
          : "Saved."
      );
      await refresh();
    } catch (err) {
      setError(err.message || "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleTestRoute = async (sendEmail = false) => {
    if (!testEmail.trim()) {
      setError("Enter a lead's email from the inbox.");
      return;
    }
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const result = sendEmail
        ? await sendRoute(testEmail.trim(), { send_email: true })
        : await fetchJson("/routing/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: testEmail.trim() }),
          });
      if (result.assigned) {
        setMessage(
          `Would go to ${result.rep?.name}` +
            (result.email_sent ? " — email sent!" : sendEmail && result.notify_error ? ` (email failed)` : "")
        );
      } else {
        setMessage(result.skipped_reason || "This lead wouldn't be routed.");
      }
      await refresh();
    } catch (err) {
      setError(err.message || "Test failed");
    } finally {
      setSaving(false);
    }
  };

  const handleSendUnrouted = async () => {
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const result = await sendUnrouted();
      const sent = (result.results || []).filter((r) => r.assigned).length;
      setMessage(`Sent ${sent} lead${sent === 1 ? "" : "s"} to your team.`);
      await refresh();
    } catch (err) {
      setError(err.message || "Batch send failed");
    } finally {
      setSaving(false);
    }
  };

  if (loading && !rules) {
    return <LoadingSkeleton rows={5} />;
  }

  const weekly = stats?.weekly?.by_rep || {};

  return (
    <>
      <div className="page-intro animate-fade-in">
        <div>
          <h1 className="page-title">Team</h1>
          <p className="page-subtitle">
            Set who gets new leads, how they&apos;re prioritized, and when reps get notified.
          </p>
        </div>
        <button type="button" className="btn" onClick={handleSave} disabled={saving}>
          {saving ? "Saving…" : "Save changes"}
        </button>
      </div>

      <Toast type="success" message={message} />
      <Toast type="error" message={error} />

      {!rules?.smtp_configured && (
        <Card className="banner-warn" delay={40}>
          <p>
            <strong>Rep notifications aren&apos;t on yet.</strong> Leads can still be assigned in the inbox.
            Turn on <strong>Email the assigned rep</strong> below once your admin finishes email setup on the Setup tab.
          </p>
        </Card>
      )}

      <Card title="When a lead comes in" subtitle="Usually within about 30 seconds of the form" delay={60}>
        <p className="field-hint" style={{ marginBottom: "1rem" }}>
          Turn on auto-assign and notifications here. Set percentages and scoring rules in the section below your team cards.
        </p>
        <Toggle
          checked={!!rules?.auto_route_enabled}
          onChange={(v) => setRules({ ...rules, auto_route_enabled: v })}
          label="Assign new leads automatically"
          description="Picks the right rep as soon as the lead is scored"
        />
        <Toggle
          checked={!!rules?.send_email_on_route}
          onChange={(v) => setRules({ ...rules, send_email_on_route: v })}
          label="Email the assigned rep"
          description="Sends a notification when email or automation is set up in Setup"
        />
        <Toggle
          checked={!!rules?.sync_hubspot_on_route}
          onChange={(v) => setRules({ ...rules, sync_hubspot_on_route: v })}
          label="Update HubSpot from this app (optional)"
          description="Off when HubSpot runs in n8n only. Needs HUBSPOT_ACCESS_TOKEN on Railway."
        />
      </Card>

      <Card
        title="Your team"
        subtitle="Name, email, and weekly limits — save when you change distribution below"
        delay={100}
      >
        <div className="team-grid">
          {(rules?.reps || []).map((rep, index) => {
            const cap = parseWeeklyCap(rep.weekly_cap);
            const count = weekly[rep.id] ?? 0;
            const capPct = cap ? Math.min(100, (count / cap) * 100) : 0;
            return (
              <div key={rep.id || index} className={`team-card ${repBucketClass(rep.bucket)} animate-slide-up`} style={{ animationDelay: `${index * 70}ms` }}>
                <div className="team-card-body">
                  <div className="team-card-head">
                    <div className="team-avatar" aria-hidden="true">
                      {rep.name?.charAt(0) || "?"}
                    </div>
                    <label className="field-label team-name-field">
                      Name
                      <input
                        className="input team-name-input"
                        value={rep.name || ""}
                        onChange={(e) => updateRep(index, "name", e.target.value)}
                        placeholder="Full name"
                      />
                    </label>
                  </div>

                  {rep.bucket !== "automation" && (
                  <label className="field-label">
                    Email
                    <input
                      className="input"
                      type="email"
                      value={rep.email || ""}
                      onChange={(e) => updateRep(index, "email", e.target.value)}
                      placeholder="rep@email.com"
                    />
                  </label>
                  )}

                  {rep.bucket !== "automation" && (
                  <label className="field-label">
                    Mobile (SMS / caller ID)
                    <input
                      className="input"
                      type="tel"
                      value={rep.phone || ""}
                      onChange={(e) => updateRep(index, "phone", e.target.value)}
                      placeholder="9083376143"
                    />
                  </label>
                  )}

                  {rep.bucket !== "automation" && (
                  <label className="field-label">
                    HubSpot owner ID
                    <input
                      className="input"
                      type="text"
                      inputMode="numeric"
                      list={hubspotOwners.length ? "hubspot-owners-list" : undefined}
                      value={rep.hubspot_owner_id ?? ""}
                      onChange={(e) => updateRep(index, "hubspot_owner_id", e.target.value.trim())}
                      placeholder="For n8n Contact owner — paste ID from HubSpot Users"
                    />
                  </label>
                  )}

                  {rep.bucket !== "automation" && (
                  <label className="field-label">
                    Weekly limit
                    <input
                      className="input cap-input"
                      type="number"
                      min="0"
                      step="1"
                      value={rep.weekly_cap ?? ""}
                      onChange={(e) =>
                        updateRep(index, "weekly_cap", e.target.value === "" ? null : e.target.value)
                      }
                      placeholder="No limit"
                    />
                  </label>
                  )}

                  {rep.bucket === "general" && (
                  <label className="field-label">
                    Lead share
                    <input
                      className="input cap-input"
                      type="number"
                      min="1"
                      step="1"
                      value={rep.weight ?? 1}
                      onChange={(e) => updateRep(index, "weight", e.target.value)}
                    />
                    <span className="field-hint">
                      Share vs. others in the same pool — 2 means twice the leads of a rep on 1
                      {sharePercent(rep, rules?.reps) != null
                        ? ` (currently ~${sharePercent(rep, rules.reps)}% of this pool)`
                        : ""}
                    </span>
                  </label>
                  )}
                </div>

                <div className="team-card-footer">
                  <div className="cap-meter">
                    <div className="cap-label">
                      This week: <strong>{count}</strong>
                      {cap != null ? ` of ${cap}` : " assigned · no limit"}
                    </div>
                    <div className="cap-track">
                      <div className="cap-fill" style={{ width: `${capPct}%` }} />
                    </div>
                  </div>

                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {wufooForms.length > 0 && (
        <Card delay={112}>
          <h3 className="card-title">Wufoo forms</h3>
          <p className="card-copy">
            Each form uses the same app webhook with a different <code>?form=</code> id. Set{" "}
            <code>field_map</code> (Wufoo → column) and <code>display_fields</code> (column → email label) in{" "}
            <code>config/wufoo_forms.json</code>. Form 1 uses AI scoring + Team rules. Other forms can use fixed
            reps (e.g. Beau / Eric).
          </p>
          <div className="dist-table-wrap">
            <table className="dist-table">
              <thead>
                <tr>
                  <th>Form</th>
                  <th>Routing</th>
                  <th>Fields</th>
                  <th>Email fields</th>
                  <th>Wufoo webhook URL</th>
                </tr>
              </thead>
              <tbody>
                {wufooForms.map((f) => (
                  <tr key={f.id}>
                    <td>
                      <strong>{f.label || f.id}</strong>
                      <div className="field-hint">{f.id}</div>
                    </td>
                    <td>{f.routing_policy === "ai" ? "AI + Team rules" : f.routing_policy === "fixed_reps" ? "Fixed reps" : f.routing_policy}</td>
                    <td>{f.field_count ?? "—"} mapped</td>
                    <td>{f.display_field_count ?? "—"} in email</td>
                    <td>
                      <code style={{ fontSize: "0.75rem", wordBreak: "break-all" }}>{f.webhook_url_example}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="field-hint" style={{ marginTop: "0.75rem" }}>
            Edit <code>config/wufoo_forms.json</code> on the server to add forms, field maps, and routing policy.
            n8n receives <code>body.form.routing_policy</code> and <code>body.form.id</code> on every assignment.
          </p>
        </Card>
      )}

      {hubspotOwners.length > 0 && (
        <datalist id="hubspot-owners-list">
          {hubspotOwners.map((o) => (
            <option key={o.id} value={o.id}>
              {o.name} ({o.email})
            </option>
          ))}
        </datalist>
      )}

      <Card className="banner-warn" delay={115}>
        <p>
          <strong>HubSpot runs in n8n</strong> — this app does not update HubSpot (leave{" "}
          <em>Update HubSpot from this app</em> off). The webhook still sends contact data to n8n. Paste each
          rep&apos;s <strong>HubSpot owner ID</strong> on their card so n8n gets{" "}
          <code>body.rep.hubspot_owner_id</code>. In your HubSpot node: <strong>Contact Owner Name or ID</strong>{" "}
          = <code>{`{{ $json.body.rep.hubspot_owner_id }}`}</code> (not <code>Number()</code> — avoids NaN when missing).
        </p>
      </Card>

      <Card
        title="Lead distribution & rules"
        subtitle="One place to compare everyone — edit percentages and scoring, then Save changes at the top"
        delay={120}
      >
        <DistributionPanel
          rules={rules}
          rubric={rubric}
          coachingBoost={coachingBoost}
          icpLlmMin={icpLlmMin}
          onRulesChange={(key, value) => setRules((prev) => ({ ...prev, [key]: value }))}
          onRubricChange={(key, value) => setRubric((prev) => ({ ...prev, [key]: value }))}
          onCoachingBoostChange={setCoachingBoost}
          onIcpLlmMinChange={setIcpLlmMin}
        />
      </Card>

      <Card title="Try it" subtitle="Pick a lead from your inbox by email" delay={140}>
        <div className="inline-form">
          <input
            className="input"
            type="email"
            placeholder="Lead email address"
            value={testEmail}
            onChange={(e) => setTestEmail(e.target.value)}
          />
          <button type="button" className="btn secondary" onClick={() => handleTestRoute(false)} disabled={saving}>
            See who would get it
          </button>
          <button type="button" className="btn" onClick={() => handleTestRoute(true)} disabled={saving}>
            Assign &amp; notify now
          </button>
        </div>
        <button type="button" className="btn ghost" onClick={handleSendUnrouted} disabled={saving}>
          Assign everyone still waiting
        </button>
      </Card>

      {stats?.recent?.length > 0 && (
        <Card title="Recent assignments" subtitle="Newest first" delay={220}>
          <ul className="assignment-list">
            {stats.recent.map((row, i) => (
              <li key={`${row.at}-${i}`} className="assignment-row animate-fade-in" style={{ animationDelay: `${i * 40}ms` }}>
                <span className="assignment-who">{row.rep_name}</span>
                <span className="assignment-lead">{row.lead_email}</span>
                <Badge tier={row.ai_tier} showEmoji={false} />
                {row.email_sent && <span className="sent-tag">Notified</span>}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </>
  );
}
