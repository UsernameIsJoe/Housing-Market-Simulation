
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from typing import Dict, List, Any
import random

@dataclass
class PersonaPrior:
    name: str
    income_mu: float
    income_sigma: float
    sf_mu: float
    sf_sigma: float
    bed_need_probs: Dict[int, float]
    floor_pref_probs: Dict[str, float]
    noise_tol_probs: Dict[str, float]
    elasticity_mean: float
    style_vocab: Dict[str, float]
    amenity_vocab: Dict[str, float]
    cultural_fit_mean: float
    share: float

def _parse_jsonish(cell: str):
    """
    Parse a JSON-like string from CSV. Expects standard JSON.
    Examples:
      {"1":0.7,"2":0.3}
      {"low":0.3,"med":0.5,"high":0.2}
    """
    if pd.isna(cell) or cell == "":
        return {}
    try:
        obj = json.loads(cell)
        # Convert numeric keys that were saved as strings to ints when appropriate
        if all(k.isdigit() for k in obj.keys()):
            obj = {int(k): v for k, v in obj.items()}
        return obj
    except Exception as e:
        raise ValueError(f"Cannot parse JSON from: {cell}") from e

def load_personas_csv(csv_path: str) -> List[PersonaPrior]:
    df = pd.read_csv(csv_path)
    required_cols = [
        "name","income_mu","income_sigma","sf_mu","sf_sigma",
        "bed_need_probs","floor_pref_probs","noise_tol_probs",
        "elasticity_mean","style_vocab","amenity_vocab",
        "cultural_fit_mean","share"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in personas CSV: {missing}")
    personas = []
    for _, row in df.iterrows():
        p = PersonaPrior(
            name=str(row["name"]),
            income_mu=float(row["income_mu"]),
            income_sigma=float(row["income_sigma"]),
            sf_mu=float(row["sf_mu"]),
            sf_sigma=float(row["sf_sigma"]),
            bed_need_probs=_parse_jsonish(row["bed_need_probs"]),
            floor_pref_probs=_parse_jsonish(row["floor_pref_probs"]),
            noise_tol_probs=_parse_jsonish(row["noise_tol_probs"]),
            elasticity_mean=float(row["elasticity_mean"]),
            style_vocab=_parse_jsonish(row["style_vocab"]),
            amenity_vocab=_parse_jsonish(row["amenity_vocab"]),
            cultural_fit_mean=float(row["cultural_fit_mean"]),
            share=float(row["share"]),
        )
        personas.append(p)
    # normalize shares
    total_share = sum(p.share for p in personas)
    for p in personas:
        p.share = p.share / total_share if total_share > 0 else 1.0/len(personas)
    return personas

def _sample_from_probs(d: Dict[Any, float], rng: random.Random):
    keys = list(d.keys())
    weights = np.array(list(d.values()), dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()
    return rng.choices(keys, weights=weights, k=1)[0]

def _weighted_choice_words(weight_map: Dict[str, float], k: int, rng: random.Random) -> List[str]:
    keys = list(weight_map.keys())
    if not keys:
        return []
    weights = np.array(list(weight_map.values()), dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights)/len(weights)
    idxs = rng.choices(range(len(keys)), weights=weights, k=k)
    # unique-ish
    seen, out = set(), []
    for i in idxs:
        if keys[i] not in seen:
            out.append(keys[i]); seen.add(keys[i])
        if len(out) >= k: break
    if len(out) < k:
        remaining = [kw for kw in keys if kw not in seen]
        rng.shuffle(remaining)
        out += remaining[:(k-len(out))]
    return out

def _clamp(x, a, b):
    return max(a, min(b, x))

def sample_agent(prior: PersonaPrior, rng: random.Random, housing_ratio_range=(0.25,0.33)) -> Dict[str, Any]:
    # Income (monthly)
    income = max(1800, rng.normalvariate(prior.income_mu, prior.income_sigma))
    lo, hi = housing_ratio_range
    housing_ratio = rng.uniform(lo, hi)
    wtp = income * housing_ratio * rng.uniform(0.95, 1.05)

    # SF band
    sf_center = max(350, rng.normalvariate(prior.sf_mu, prior.sf_sigma))
    sf_low = _clamp(sf_center - rng.uniform(50, 120), 320, 1800)
    sf_high = _clamp(sf_center + rng.uniform(50, 150), sf_low + 30, 2200)

    # Categorical prefs
    bed_need = int(_sample_from_probs(prior.bed_need_probs, rng)) if prior.bed_need_probs else 1
    floor_pref = _sample_from_probs(prior.floor_pref_probs, rng) if prior.floor_pref_probs else "med"
    noise_tol = _sample_from_probs(prior.noise_tol_probs, rng) if prior.noise_tol_probs else "med"

    elasticity = _clamp(rng.normalvariate(prior.elasticity_mean, 0.1), 0.4, 1.6)
    style_words = _weighted_choice_words(prior.style_vocab, k=3, rng=rng) if prior.style_vocab else []
    amenity_rank = _weighted_choice_words(prior.amenity_vocab, k=3, rng=rng) if prior.amenity_vocab else []
    cultural_fit = _clamp(random.gauss(prior.cultural_fit_mean, 0.15), 0, 1)

    availability_month = int(np.random.geometric(p=0.35))  # 1,2,3...

    return {
        "persona_base": prior.name,
        "income_monthly": round(income, 0),
        "housing_ratio": round(housing_ratio, 3),
        "wtp_monthly": int(round(wtp, 0)),
        "bed_need": bed_need,
        "sf_pref_low": int(round(sf_low, 0)),
        "sf_pref_high": int(round(sf_high, 0)),
        "floor_pref": floor_pref,
        "noise_tol": noise_tol,
        "elasticity": round(elasticity, 2),
        "style_words": style_words,
        "amenity_rank": amenity_rank,
        "cultural_fit": round(cultural_fit, 2),
        "availability_month": availability_month
    }

def generate_agents_from_csv(
    personas_csv: str,
    n_agents: int = 500,
    use_mixture_personas: bool = False,
    dirichlet_alpha: List[float] = None,
    housing_ratio_ranges_by_persona: Dict[str, tuple] = None,
    seed: int = 123
) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    personas = load_personas_csv(personas_csv)
    name_to_persona = {p.name: p for p in personas}
    persona_names = [p.name for p in personas]

    # housing ratio defaults per persona if not provided
    if housing_ratio_ranges_by_persona is None:
        housing_ratio_ranges_by_persona = {name: (0.25, 0.33) for name in persona_names}

    agents = []
    if use_mixture_personas:
        if dirichlet_alpha is None:
            dirichlet_alpha = [2.0] * len(personas)
        for i in range(n_agents):
            w = np.random.dirichlet(alpha=dirichlet_alpha)
            # Blend a temporary prior
            def blend_prob_map(attr):
                keys = set().union(*[set(getattr(p, attr).keys()) for p in personas])
                probs = {k: 0.0 for k in keys}
                for idx, p in enumerate(personas):
                    for k, v in getattr(p, attr).items():
                        probs[k] += w[idx] * v
                total = sum(probs.values())
                if total <= 0:
                    return {k: 1.0/len(probs) for k in probs}
                return {k: v/total for k, v in probs.items()}

            def blend_vocab(attr):
                keys = set().union(*[set(getattr(p, attr).keys()) for p in personas])
                wm = {k:0.0 for k in keys}
                for idx, p in enumerate(personas):
                    for k, v in getattr(p, attr).items():
                        wm[k] += w[idx]*v
                total = sum(max(0.0, v) for v in wm.values())
                if total <= 0:
                    return {k:1.0/len(wm) for k in wm}
                return {k: max(0.0, v)/total for k, v in wm.items()}

            temp = PersonaPrior(
                name="mixture",
                income_mu=sum(w[i]*personas[i].income_mu for i in range(len(personas))),
                income_sigma=sum(w[i]*personas[i].income_sigma for i in range(len(personas))),
                sf_mu=sum(w[i]*personas[i].sf_mu for i in range(len(personas))),
                sf_sigma=sum(w[i]*personas[i].sf_sigma for i in range(len(personas))),
                bed_need_probs=blend_prob_map("bed_need_probs"),
                floor_pref_probs=blend_prob_map("floor_pref_probs"),
                noise_tol_probs=blend_prob_map("noise_tol_probs"),
                elasticity_mean=sum(w[i]*personas[i].elasticity_mean for i in range(len(personas))),
                style_vocab=blend_vocab("style_vocab"),
                amenity_vocab=blend_vocab("amenity_vocab"),
                cultural_fit_mean=sum(w[i]*personas[i].cultural_fit_mean for i in range(len(personas))),
                share=1.0
            )
            # choose housing ratio from dominant base persona
            dominant = persona_names[int(np.argmax(w))]
            agent = sample_agent(temp, rng, housing_ratio_range=housing_ratio_ranges_by_persona.get(dominant, (0.25,0.33)))
            agent["persona_mixture"] = {persona_names[i]: float(round(w[i],3)) for i in range(len(personas))}
            agent["persona_base"] = dominant
            agent["agent_id"] = f"A{i+1:04d}"
            agents.append(agent)
    else:
        # Hard persona sampling according to shares
        shares = np.array([p.share for p in personas], dtype=float)
        shares = shares / shares.sum() if shares.sum() > 0 else np.ones(len(personas))/len(personas)
        counts = np.random.multinomial(n_agents, shares)
        idx = 0
        for p_idx, prior in enumerate(personas):
            for _ in range(counts[p_idx]):
                agent = sample_agent(prior, rng, housing_ratio_range=housing_ratio_ranges_by_persona.get(prior.name, (0.25,0.33)))
                idx += 1
                agent["persona_mixture"] = None
                agent["agent_id"] = f"A{idx:04d}"
                agents.append(agent)

    df = pd.DataFrame(agents)
    # Nice column order
    col_order = [
        "agent_id","persona_base","persona_mixture",
        "income_monthly","housing_ratio","wtp_monthly",
        "bed_need","sf_pref_low","sf_pref_high",
        "floor_pref","noise_tol","elasticity",
        "cultural_fit","availability_month",
        "style_words","amenity_rank"
    ]
    df = df[col_order]
    return df
