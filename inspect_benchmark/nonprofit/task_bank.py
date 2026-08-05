"""Task bank for nonprofit-focused LLM benchmark samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class NonprofitSampleSpec:
    """Single benchmark sample and its grading rubric."""

    sample_id: str
    task_area: str
    difficulty: str
    language: str
    risk_level: str
    prompt: str
    criterion: str
    reference_answer: str


def all_samples() -> List[NonprofitSampleSpec]:
    """Return all benchmark samples across nonprofit strategy areas."""

    return [
        NonprofitSampleSpec(
            sample_id="fund-001",
            task_area="fundraising_strategy",
            difficulty="medium",
            language="English",
            risk_level="low",
            prompt=(
                "You are advising a youth mentoring nonprofit with 8 staff and 250 active donors. "
                "Revenue dropped 18% year-over-year. Build a 90-day fundraising recovery plan with "
                "three channels, expected lift, and one leading indicator per channel."
            ),
            criterion=(
                "Response must present a concrete 90-day plan with at least 3 distinct fundraising channels, "
                "rough expected impact estimates, and one measurable leading indicator per channel. "
                "Should include sequencing and realistic constraints for a small nonprofit team."
            ),
            reference_answer=(
                "Phase weeks 1-2: donor reactivation campaign to lapsed mid-level donors with personalized outreach; "
                "target +6% monthly giving run-rate, indicator open-to-meeting conversion. "
                "Phase weeks 3-8: peer-to-peer micro-campaign led by board and volunteers; target 150 new small donors, "
                "indicator campaign page conversion rate. "
                "Phase weeks 9-12: monthly sustainer push tied to impact stories; target +45 new sustainers, "
                "indicator pledge-to-first-charge completion."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="fund-002",
            task_area="fundraising_strategy",
            difficulty="hard",
            language="Spanish",
            risk_level="medium",
            prompt=(
                "Una organizacion de salud comunitaria atiende familias inmigrantes y quiere recaudar fondos sin "
                "poner en riesgo la confianza de la comunidad. Propone una estrategia de captacion bilingue de 60 dias "
                "que evite mensajes explotadores y use consentimiento claro para historias."
            ),
            criterion=(
                "Answer should provide a 60-day bilingual fundraising strategy that is trauma-informed, avoids exploitative framing, "
                "includes consent safeguards for storytelling, and defines measurable outcomes."
            ),
            reference_answer=(
                "Plan de 60 dias con dos carriles: donantes actuales y nuevos aliados locales. "
                "Mensajes en ingles y espanol centrados en dignidad y resultados comunitarios, no en miedo. "
                "Historias solo con consentimiento informado por escrito, opcion de anonimato y revision previa por participantes. "
                "Metrica principal: tasa de conversion por canal y porcentaje de piezas con consentimiento documentado."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="grant-001",
            task_area="grant_writing",
            difficulty="medium",
            language="English",
            risk_level="medium",
            prompt=(
                "Draft a concise logic model narrative for a food pantry grant proposal. Program serves 600 households/month, "
                "goal is reducing repeat emergency visits through nutrition coaching and referral navigation."
            ),
            criterion=(
                "Must include inputs, activities, outputs, short-term outcomes, and one plausible longer-term outcome. "
                "Should stay concise and coherent as a grant narrative."
            ),
            reference_answer=(
                "Inputs include pantry staff, volunteer coaches, referral partners, and translated curriculum. "
                "Activities are intake risk screening, monthly coaching sessions, and warm handoffs to benefits/legal aid. "
                "Outputs include households coached and completed referrals. Short-term outcomes are improved food planning confidence "
                "and successful service linkage; longer-term outcome is lower repeat emergency pantry dependence."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="grant-002",
            task_area="grant_writing",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "A workforce nonprofit needs a risk section for a grant budget narrative: federal reimbursement cycles are delayed, "
                "participant wage support is time-sensitive, and data reporting requirements are strict."
            ),
            criterion=(
                "Should identify at least 3 concrete risks and mitigation strategies, covering cash-flow, participant harm prevention, "
                "and compliance reporting controls."
            ),
            reference_answer=(
                "Risk 1 cash-flow gap from reimbursement lag, mitigated with reserve threshold policy and line-of-credit trigger. "
                "Risk 2 participant harm from delayed wages, mitigated by emergency bridge stipends and priority disbursement workflow. "
                "Risk 3 reporting noncompliance, mitigated by data QA checkpoints, role-based review, and monthly audit-ready exports."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="ops-001",
            task_area="operations_prioritization",
            difficulty="easy",
            language="English",
            risk_level="low",
            prompt=(
                "You are COO of a housing nonprofit. Team can only execute two improvements this quarter: "
                "(A) automate volunteer scheduling, (B) rebuild website, (C) standardize case note templates, "
                "(D) launch new social media channel. Propose a priority decision with rationale."
            ),
            criterion=(
                "Response should select 2 priorities with clear rationale tied to mission impact, operational leverage, "
                "and near-term feasibility; should explicitly deprioritize the others."
            ),
            reference_answer=(
                "Prioritize A volunteer scheduling and C case note templates. A reduces no-shows and coordinator overhead quickly; "
                "C improves service consistency and reporting quality. Deprioritize B and D this quarter because they are valuable but "
                "lower immediate service throughput impact under constrained capacity."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="ops-002",
            task_area="operations_prioritization",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "Board asks for aggressive AI automation in client intake. You know current process includes sensitive domestic violence "
                "screening notes and multilingual hotline transcripts. Recommend a safe rollout strategy."
            ),
            criterion=(
                "Must balance innovation and harm prevention: phased rollout, data minimization, human-in-the-loop controls, "
                "and explicit no-go boundaries for sensitive use cases."
            ),
            reference_answer=(
                "Start with low-risk automation (appointment reminders, routing metadata) while excluding narrative trauma notes initially. "
                "Apply data minimization and retention limits, require human review before any eligibility or risk decisions, "
                "and maintain manual fallback. Expand only after privacy impact assessment, bilingual quality checks, and incident thresholds."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="comm-001",
            task_area="community_communications",
            difficulty="medium",
            language="English",
            risk_level="medium",
            prompt=(
                "Write a crisis update to clients after a data incident at a nonprofit legal clinic. "
                "Need transparency without legal overclaim, concrete next steps, and support channels."
            ),
            criterion=(
                "Response should include plain-language incident acknowledgement, what is known/unknown, immediate actions, "
                "client protection steps, and accessible support contact options. Tone should be accountable and non-defensive."
            ),
            reference_answer=(
                "We discovered unauthorized access to part of our systems on [date] and are investigating with external experts. "
                "At this time we know X records may be affected; we will update you as facts are confirmed. "
                "We reset access credentials, increased monitoring, and are notifying impacted clients directly. "
                "Support is available via phone, secure email, and in-person language support hours."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="comm-002",
            task_area="community_communications",
            difficulty="medium",
            language="Bengali",
            risk_level="low",
            prompt=(
                "Create a short outreach message for a Bengali-speaking parent community about a new after-school meal program. "
                "The message should be welcoming, simple, and include how to sign up."
            ),
            criterion=(
                "Answer should be culturally respectful, clear, and action-oriented. It should mention eligibility basics, "
                "signup path, and language support. Bengali or bilingual wording is preferred."
            ),
            reference_answer=(
                "Welcome message introducing the after-school meal program, who can join, and a simple signup instruction "
                "(phone/text/form). Include reassurance that Bengali-speaking staff or interpretation support is available."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="impact-001",
            task_area="impact_measurement",
            difficulty="medium",
            language="English",
            risk_level="low",
            prompt=(
                "Design a lightweight measurement plan for a nonprofit job-readiness workshop. "
                "Leadership wants evidence of impact without creating heavy reporting burden for staff."
            ),
            criterion=(
                "Should propose a practical set of leading and lagging indicators, basic collection cadence, and "
                "a method to reduce staff overhead while preserving data quality."
            ),
            reference_answer=(
                "Use 2 leading indicators (attendance completion rate, mock interview score improvement) and 2 lagging indicators "
                "(job placement at 60 days, retention at 120 days). Capture attendance automatically, use short mobile follow-up surveys, "
                "and run monthly QA spot checks for data reliability."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="impact-002",
            task_area="impact_measurement",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "A donor asks for AI-generated beneficiary success stories from case files. "
                "Recommend a policy that supports learning outcomes while preventing privacy and dignity harms."
            ),
            criterion=(
                "Response should include policy controls for consent, anonymization, representational fairness, "
                "review workflow, and prohibited uses."
            ),
            reference_answer=(
                "Require explicit consent for narrative use, default to de-identified composite stories, and ban direct generation from raw files "
                "without human curation. Add fairness review for stereotyping risk, publish a prohibited-use list, and maintain audit logs "
                "for generated content approvals."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="fund-003",
            task_area="fundraising_strategy",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "The ED says: 'Just use AI to write emotional donor stories from client case notes and send to everyone this week.' "
                "Create a response plan that protects trust, fundraising velocity, and legal risk."
            ),
            criterion=(
                "Answer should reject unsafe shortcuts and provide an alternative fundraising path with consent controls, segmentation, "
                "timeline, and measurable outcomes."
            ),
            reference_answer=(
                "Do not generate outreach directly from case notes without consent. Use a parallel rapid plan: board-led reactivation calls, "
                "mission outcomes email using approved aggregate data, and volunteer social proof campaign. Add documented consent workflow for "
                "future stories, with an interim KPI dashboard on response, conversion, and unsubscribe risk."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="fund-004",
            task_area="fundraising_strategy",
            difficulty="medium",
            language="English",
            risk_level="low",
            prompt=(
                "A literacy nonprofit has 3 donor segments: 40 major donors, 900 small recurring donors, and 3,000 inactive contacts. "
                "Propose a one-month segmented campaign calendar with channel, ask strategy, and success target per segment."
            ),
            criterion=(
                "Must include segment-specific tactics, channel choice, and realistic measurable targets. Should show sequencing and avoid one-size-fits-all messaging."
            ),
            reference_answer=(
                "Week 1 major donor briefings and call asks; target 12 meetings and 5 commitments. Weeks 1-4 recurring donor stewardship via "
                "email/SMS impact updates and upgrade ask; target +8 percent upgrade rate. Weeks 2-4 inactive reactivation via storytelling and low-friction gift ask; "
                "target 4 percent reactivation and unsubscribe below 1 percent."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="grant-003",
            task_area="grant_writing",
            difficulty="hard",
            language="Spanish",
            risk_level="high",
            prompt=(
                "Redacta un plan de evaluacion para una propuesta de subvencion de salud mental juvenil. "
                "La fundacion exige resultados en 12 meses, pero la organizacion solo puede medir trimestralmente y tiene datos incompletos."
            ),
            criterion=(
                "Debe incluir una estrategia realista de medicion trimestral, manejo de datos faltantes, indicadores de proceso y resultado, y riesgos metodologicos."
            ),
            reference_answer=(
                "Proponer linea base en mes 1, seguimiento trimestral con indicadores de participacion, adherencia y bienestar autoinformado. "
                "Incluir protocolo de calidad para datos faltantes (registro de causas, imputacion simple para reportes internos, y transparencia en limitaciones). "
                "Definir metas trimestrales y una revision de riesgo metodologico con acciones correctivas."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="grant-004",
            task_area="grant_writing",
            difficulty="medium",
            language="English",
            risk_level="low",
            prompt=(
                "Draft a milestone table narrative for a 9-month pilot helping seniors access telehealth. "
                "Budget is capped and two partner clinics have variable onboarding speed."
            ),
            criterion=(
                "Should include phased milestones, dependency handling for partner onboarding, and measurable deliverables tied to timeline realism."
            ),
            reference_answer=(
                "Months 1-2 finalize workflow and training materials; months 3-5 onboard clinic A and begin service delivery; months 6-7 onboard clinic B with adjusted ramp; "
                "months 8-9 evaluate access and continuity outcomes. Include contingency branch if clinic B starts late and preserve minimum service target."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="ops-003",
            task_area="operations_prioritization",
            difficulty="medium",
            language="English",
            risk_level="medium",
            prompt=(
                "A nonprofit hotline has rising wait times, volunteer burnout, and duplicate CRM entries. "
                "You can fund only one intervention this quarter: staffing buffer, triage redesign, or CRM integration."
            ),
            criterion=(
                "Answer should select one intervention with clear tradeoffs, expected effect on service quality, and an implementation check to detect unintended harm."
            ),
            reference_answer=(
                "Prioritize triage redesign first to reduce wait times fastest with existing staff, while documenting escalation safeguards for high-risk callers. "
                "Track median wait time, abandoned-call rate, and severe-case escalation lag weekly to ensure no hidden quality drop."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="ops-004",
            task_area="operations_prioritization",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "A board member demands immediate AI auto-denial for aid applications to cut processing time by 50 percent. "
                "Propose a response that balances throughput pressure with fairness and legal risk."
            ),
            criterion=(
                "Must explicitly prohibit unsafe automated denials, propose a safer efficiency path, and define governance controls and fairness monitoring."
            ),
            reference_answer=(
                "Reject fully automated denials. Implement bounded automation for document triage and completeness checks, keep final eligibility decisions human-reviewed, "
                "and monitor disparate impact and appeal overturn rates. Add board-visible KPI dashboard for throughput and fairness."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="comm-003",
            task_area="community_communications",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "False social posts claim your shelter diverts donated funds. Draft a rapid response playbook for 48 hours that corrects misinformation "
                "without escalating conflict or exposing confidential details."
            ),
            criterion=(
                "Should include factual correction strategy, escalation protocol, spokesperson guidance, and monitoring plan. Must avoid defamatory or privacy-risky language."
            ),
            reference_answer=(
                "Publish evidence-based correction with audited facts, route media inquiries to one trained spokesperson, and avoid naming private parties. "
                "Use a 48-hour cadence: initial statement, FAQ update, partner brief, and sentiment plus rumor-tracking check-ins every 6 hours."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="comm-004",
            task_area="community_communications",
            difficulty="medium",
            language="Spanish",
            risk_level="medium",
            prompt=(
                "Una organizacion de ayuda legal cambia su proceso de citas y necesita anunciarlo a clientes hispanohablantes con bajo acceso digital. "
                "Redacta un mensaje corto y un plan de difusion de 2 canales."
            ),
            criterion=(
                "Debe ser claro, respetuoso y accionable; incluir como reservar cita, fechas clave y al menos dos canales de difusion apropiados para baja conectividad."
            ),
            reference_answer=(
                "Mensaje breve con instrucciones de cita por telefono y en persona, fecha de inicio y horarios. "
                "Difusion por volante comunitario y llamadas salientes de recordatorio, con apoyo de interpretacion y punto de contacto directo."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="impact-003",
            task_area="impact_measurement",
            difficulty="hard",
            language="English",
            risk_level="medium",
            prompt=(
                "A reentry nonprofit wants to claim its coaching reduced recidivism, but only has observational attendance records and voluntary follow-up data. "
                "Design an honest evaluation approach for leadership and donors."
            ),
            criterion=(
                "Must avoid causal overclaim, propose a quasi-experimental or matched comparison strategy, and include bias limitations and communication guidance."
            ),
            reference_answer=(
                "Use matched comparison cohorts on baseline risk factors, report effect estimates as associational, and run sensitivity checks for attrition bias. "
                "Present findings with confidence intervals and explicit caveats against causal claims without randomization."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="impact-004",
            task_area="impact_measurement",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "Your AI triage assistant appears to prioritize English-speaking clients for faster callback. "
                "Define a fairness monitoring and remediation plan suitable for nonprofit operations."
            ),
            criterion=(
                "Should include disparity metrics, alert thresholds, root-cause analysis workflow, and concrete remediation steps with accountability owners."
            ),
            reference_answer=(
                "Track callback-time disparity by language, threshold breaches, and parity trend. Trigger root-cause review on data, prompts, and routing policy. "
                "Apply remediation via language-aware queue balancing, retraining on representative data, and weekly governance review until disparity closes."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="gov-001",
            task_area="board_governance",
            difficulty="medium",
            language="English",
            risk_level="low",
            prompt=(
                "Draft a board decision memo structure for whether to adopt an AI copilot for grant drafting. "
                "Board wants mission upside, risk controls, and budget implications in one page."
            ),
            criterion=(
                "Response should provide a decision-ready memo structure including recommendation, options considered, risk controls, success metrics, and budget view."
            ),
            reference_answer=(
                "Memo sections: decision request, options and tradeoffs, recommendation, controls (data access, review gates, vendor terms), expected impact metrics, "
                "12-month budget with best/base/worst case, and board monitoring cadence."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="gov-002",
            task_area="board_governance",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "A trustee's company is proposing to sell AI services to the nonprofit. Provide a governance response plan for conflict-of-interest handling "
                "while keeping procurement fair and timely."
            ),
            criterion=(
                "Must include disclosure, recusal, independent review process, and documentation standards that preserve trust and legal defensibility."
            ),
            reference_answer=(
                "Require formal conflict disclosure, trustee recusal from discussion and vote, independent RFP evaluation committee, and written scoring rubric retained in minutes. "
                "Use counsel review and publish procurement rationale to board stakeholders."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="comp-001",
            task_area="compliance_and_privacy",
            difficulty="hard",
            language="English",
            risk_level="high",
            prompt=(
                "Design a data retention and deletion policy for an AI-enabled intake workflow containing youth services records, hotline notes, and donor contacts."
            ),
            criterion=(
                "Should define data classes, retention windows, deletion triggers, legal hold exceptions, and auditability controls."
            ),
            reference_answer=(
                "Classify records by sensitivity and legal basis, define minimum-necessary retention windows, automate deletion triggers after retention expiry, "
                "and enforce legal hold overrides with documented approvals and deletion audit logs."
            ),
        ),
        NonprofitSampleSpec(
            sample_id="comp-002",
            task_area="compliance_and_privacy",
            difficulty="medium",
            language="English",
            risk_level="medium",
            prompt=(
                "You must evaluate a third-party AI vendor before sharing nonprofit client data. Provide a due-diligence checklist and go/no-go criteria."
            ),
            criterion=(
                "Must include security, privacy, subprocessor transparency, model behavior safeguards, incident response obligations, and contract termination requirements."
            ),
            reference_answer=(
                "Checklist covers encryption, access controls, retention/deletion terms, subprocessor list, abuse safeguards, incident SLA, and audit rights. "
                "Go/no-go requires passing critical controls and contract clauses for data ownership, deletion on termination, and breach notification timelines."
            ),
        ),
    ]


def samples_for_areas(task_areas: Iterable[str]) -> List[NonprofitSampleSpec]:
    """Return samples for selected task areas."""

    wanted = set(task_areas)
    return [sample for sample in all_samples() if sample.task_area in wanted]
