# -*- coding: utf-8 -*-
"""
Interface Text Mining (Flask) :
  1) NER -> sortie BIO        2) Classification        3) Recherche TF-IDF (cosinus)
  4) RAG avec choix du LLM (ChatGPT / DeepSeek)        5) Métriques (cross-validation)
Lancer :  python app.py   puis ouvrir http://127.0.0.1:5000
"""
import os, json, re, joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.metrics.pairwise import linear_kernel

from outils_texte import normaliser, prep_recherche      # nécessaire pour recharger les modèles
from preannotation_bio import TOKEN
from train_ner import phrase_features

app = Flask(__name__)
CACHE = {}


def charger(chemin):
    """Charge un modèle depuis son chemin (le chemin est modifiable dans l'interface)."""
    chemin = os.path.abspath(chemin)
    if chemin not in CACHE:
        CACHE[chemin] = joblib.load(chemin)
    return CACHE[chemin]


# ------------------------------------------------------------------ NER
@app.post("/api/ner")
def api_ner():
    d = request.get_json()
    texte, chemin = d.get("texte", ""), d.get("chemin", "models/ner_crf.joblib")
    crf = charger(chemin)
    lignes = []
    for ligne in texte.splitlines():
        toks = [m.group() for m in TOKEN.finditer(ligne)]
        if not toks:
            continue
        tags = crf.predict_single(phrase_features(toks))
        lignes.append(list(zip(toks, tags)))
    return jsonify({"resultat": lignes})


# ------------------------------------------------------------------ Classification
@app.post("/api/classif")
def api_classif():
    d = request.get_json()
    texte, chemin = d.get("texte", ""), d.get("chemin", "models/clf_nb.joblib")
    pipe = charger(chemin)
    x = re.sub(r"\d+", " ", normaliser(texte))
    proba = pipe.predict_proba([x])[0]
    classes = list(pipe.classes_)
    return jsonify({"classe": classes[int(np.argmax(proba))],
                    "probabilites": {c: float(p) for c, p in zip(classes, proba)}})


# ------------------------------------------------------------------ Recherche TF-IDF
def chercher(requete, k=5, chemin="models/tfidf_index.joblib"):
    idx = charger(chemin)
    q = idx["vec"].transform([requete])
    sims = linear_kernel(q, idx["matrice"]).ravel()
    top = sims.argsort()[::-1][:k]
    return [{"fichier": idx["noms"][i], "classe": idx["classes"][i], "score": float(sims[i]),
             "extrait": idx["docs"][i][:600], "texte": idx["docs"][i]} for i in top if sims[i] > 0]


@app.post("/api/recherche")
def api_recherche():
    d = request.get_json()
    res = chercher(d.get("requete", ""), int(d.get("k", 5)))
    for r in res:
        r.pop("texte")
    return jsonify({"resultats": res})


# ------------------------------------------------------------------ RAG + choix du LLM
LLMS = {
    "ChatGPT": {"url": "https://api.openai.com/v1/chat/completions", "modele": "gpt-4o-mini", "env": "OPENAI_API_KEY"},
    "DeepSeek": {"url": "https://api.deepseek.com/chat/completions", "modele": "deepseek-chat", "env": "DEEPSEEK_API_KEY"},
}


@app.post("/api/rag")
def api_rag():
    import requests
    d = request.get_json()
    question, nom = d.get("question", ""), d.get("llm", "ChatGPT")
    cfg = LLMS[nom]
    cle = d.get("cle") or os.environ.get(cfg["env"], "")
    if not cle:
        return jsonify({"erreur": f"Clé API manquante : définis la variable {cfg['env']} ou colle la clé dans l'interface."}), 400
    sources = chercher(question, k=3)
    contexte = "\n\n".join(f"[{i+1}] ({s['fichier']})\n{s['texte'][:1800]}" for i, s in enumerate(sources))
    consigne = ("Tu es un assistant juridique marocain. Réponds UNIQUEMENT à partir des décisions fournies. "
                "Si l'information n'y est pas, dis-le. Cite les sources avec [1], [2], [3]. Réponds dans la langue de la question.")
    try:
        r = requests.post(cfg["url"], timeout=90,
                          headers={"Authorization": f"Bearer {cle}", "Content-Type": "application/json"},
                          json={"model": cfg["modele"], "temperature": 0,
                                "messages": [{"role": "system", "content": consigne},
                                             {"role": "user", "content": f"Décisions :\n{contexte}\n\nQuestion : {question}"}]})
        r.raise_for_status()
        reponse = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return jsonify({"erreur": f"Appel {nom} échoué : {e}"}), 502
    return jsonify({"reponse": reponse, "sources": [{"fichier": s["fichier"], "score": s["score"]} for s in sources]})


# ------------------------------------------------------------------ Métriques
@app.get("/api/metriques")
def api_metriques():
    out = {}
    for nom, f in (("ner", "metrics/ner_metrics.json"), ("classification", "metrics/clf_metrics.json")):
        if os.path.exists(f):
            out[nom] = json.load(open(f, encoding="utf-8"))
    return jsonify(out)


# ------------------------------------------------------------------ Page
PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Text Mining – Cour de cassation</title>
<style>
body{font-family:system-ui,Arial;margin:0;background:#f5f6f8;color:#222}
header{background:#1f3a5f;color:#fff;padding:14px 24px;font-size:20px}
nav{display:flex;gap:4px;padding:10px 24px 0}
nav button{border:0;padding:10px 18px;background:#dde3ec;cursor:pointer;border-radius:8px 8px 0 0;font-size:15px}
nav button.on{background:#fff;font-weight:600}
section{display:none;background:#fff;margin:0 24px 24px;padding:20px;border-radius:0 8px 8px 8px}
section.on{display:block}
textarea,input,select{width:100%;box-sizing:border-box;padding:8px;font-size:15px;margin:4px 0 10px}
textarea{min-height:120px;direction:rtl;font-family:inherit}
button.act{background:#1f3a5f;color:#fff;border:0;padding:9px 18px;border-radius:6px;cursor:pointer;font-size:15px}
table{border-collapse:collapse;margin-top:10px}td,th{border:1px solid #ccc;padding:4px 10px;text-align:center}
.tok{display:inline-block;margin:2px;padding:2px 6px;border-radius:5px;background:#eee;direction:rtl}
.DATE{background:#ffe08a}.ARTICLE{background:#b8e0ff}.CODE_LOI{background:#d5c2ff}.NUM_DOSSIER{background:#b6f0c4}.NUM_DECISION{background:#ffc2c2}
.res{border:1px solid #ddd;border-radius:6px;padding:10px;margin:8px 0}.res p{direction:rtl;margin:6px 0;color:#444}
.err{color:#b00020}small{color:#666}
</style></head><body>
<header>Text Mining – Décisions de la Cour de cassation (Chambre immobilière / Statut personnel)</header>
<nav id="nav"></nav>

<section id="ner"><h3>NER : texte → étiquettes BIO</h3>
<small>Chemin du modèle (à changer selon le modèle)</small><input id="ner_path" value="models/ner_crf.joblib">
<textarea id="ner_txt">الصادر بتاريخ 04 يناير 2022
في الملف الشرعي رقم 2021/2/2/94
وطبقا للمادة 235 من مدونة الحقوق العينية</textarea>
<button class="act" onclick="ner()">Extraire</button><div id="ner_out"></div></section>

<section id="clf"><h3>Classification</h3>
<small>Chemin du modèle</small><input id="clf_path" value="models/clf_nb.joblib">
<textarea id="clf_txt"></textarea><button class="act" onclick="clf()">Classer</button><div id="clf_out"></div></section>

<section id="rch"><h3>Recherche TF-IDF (similarité cosinus, sans IA)</h3>
<input id="rch_q" placeholder="requête en arabe"><button class="act" onclick="rch()">Chercher</button><div id="rch_out"></div></section>

<section id="rag"><h3>RAG : répondre à partir des décisions</h3>
<small>LLM</small><select id="rag_llm"><option>ChatGPT</option><option>DeepSeek</option></select>
<small>Clé API (facultatif si la variable d'environnement existe)</small><input id="rag_key" type="password">
<input id="rag_q" placeholder="question"><button class="act" onclick="rag()">Répondre</button><div id="rag_out"></div></section>

<section id="met"><h3>Métriques (validation croisée 5 plis)</h3><div id="met_out"></div></section>

<script>
const tabs={ner:"NER",clf:"Classification",rch:"Recherche",rag:"RAG / LLM",met:"Métriques"};
const nav=document.getElementById('nav');
for(const k in tabs){const b=document.createElement('button');b.textContent=tabs[k];b.onclick=()=>show(k);b.id='b_'+k;nav.appendChild(b)}
function show(k){for(const t in tabs){document.getElementById(t).classList.toggle('on',t==k);document.getElementById('b_'+t).classList.toggle('on',t==k)}if(k=='met')met()}
show('ner');
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function post(u,d){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});return [r.ok,await r.json()]}
async function ner(){const [ok,d]=await post('/api/ner',{texte:ner_txt.value,chemin:ner_path.value});
 if(!ok){ner_out.innerHTML='<p class=err>'+esc(JSON.stringify(d))+'</p>';return}
 let h='<p>';let rows='<table><tr><th>Token</th><th>Étiquette BIO</th></tr>';
 for(const l of d.resultat){for(const [t,g] of l){h+=`<span class="tok ${g.slice(2)}">${esc(t)}</span>`;if(g!='O')rows+=`<tr><td>${esc(t)}</td><td>${g}</td></tr>`}h+='<br>'}
 ner_out.innerHTML=h+'</p>'+rows+'</table><small>Seuls les tokens étiquetés (≠ O) sont listés.</small>'}
async function clf(){const [ok,d]=await post('/api/classif',{texte:clf_txt.value,chemin:clf_path.value});
 clf_out.innerHTML=ok?`<p><b>${d.classe}</b></p>`+Object.entries(d.probabilites).map(([c,p])=>`${c} : ${(p*100).toFixed(1)} %`).join('<br>'):'<p class=err>'+esc(JSON.stringify(d))+'</p>'}
async function rch(){const [ok,d]=await post('/api/recherche',{requete:rch_q.value,k:5});
 rch_out.innerHTML=ok?d.resultats.map(r=>`<div class=res><b>${r.fichier}</b> – score ${r.score.toFixed(3)}<p>${esc(r.extrait)}…</p></div>`).join('')||'Aucun résultat':'<p class=err>erreur</p>'}
async function rag(){rag_out.textContent='…';const [ok,d]=await post('/api/rag',{question:rag_q.value,llm:rag_llm.value,cle:rag_key.value});
 rag_out.innerHTML=ok?`<div class=res><p>${esc(d.reponse)}</p></div><small>Sources : ${d.sources.map(s=>s.fichier).join(', ')}</small>`:'<p class=err>'+esc(d.erreur||'erreur')+'</p>'}
async function met(){const d=await (await fetch('/api/metriques')).json();let h='';
 for(const [n,m] of Object.entries(d)){h+=`<h4>${n}</h4><table><tr><th>Métrique</th><th>Moyenne</th><th>Écart-type</th></tr>`;
  const mo=m.moyenne_ecart_type||m.cross_validation;for(const [k,v] of Object.entries(mo)){if(Array.isArray(v))h+=`<tr><td>${k}</td><td>${v[0].toFixed(3)}</td><td>${v[1].toFixed(3)}</td></tr>`}h+='</table>'}
 met_out.innerHTML=h}
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(debug=True)
