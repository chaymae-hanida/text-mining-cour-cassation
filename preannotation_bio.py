# -*- coding: utf-8 -*-
"""
Étape 1 du NER : pré-annotation automatique au format BIO (CoNLL).
- Tire 100 décisions (50 par chambre) avec une graine fixe.
- Annote 5 entités avec des règles (regex). TU CORRIGES ENSUITE À LA MAIN.
Sortie : annotations/pre_annotation.conll  (token<TAB>étiquette, ligne vide entre phrases)
"""
import re, glob, os, random
from collections import Counter

ENTITES = ["NUM_DECISION", "DATE", "NUM_DOSSIER", "ARTICLE", "CODE_LOI"]
MOIS = "يناير|فبراير|مارس|أبريل|ابريل|ماي|مايو|يونيو|يوليوز|يوليو|غشت|شتنبر|أكتوبر|اكتوبر|نونبر|دجنبر"

# (entité, regex, groupe à annoter)
MOT = r"(?!المؤرخ|الصادر|المحكمة|المطلوب|الطاعن|الشرعي|المذكور)ال\S+"
REGLES = [
    ("NUM_DECISION", re.compile(r"^(?:القرار\s+)?(?:رقم|رتم)\s+([\d/]+)\s*$", re.M), 1),
    ("NUM_DOSSIER", re.compile(r"الملف\s+(?:\S+\s+)?(?:رقم|عدد)\s+(\d{2,4}(?:/\d+){2,})"), 1),
    ("DATE", re.compile(rf"(\d{{1,2}}\s+(?:{MOIS})\s+\d{{4}})"), 1),
    ("DATE", re.compile(r"(?<![\d/])(\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}/\d{4})(?![\d/])"), 1),
    ("ARTICLE", re.compile(r"((?:المادة|المادتين|الفصل|الفصلين|المواد|الفصول)\s+\d+(?:\s*مكرر)?)"), 1),
    ("CODE_LOI", re.compile(rf"((?:مدونة|قانون|ظهير)\s+(?:الالتزامات\s+والعقود|{MOT}(?:\s+{MOT}){{0,2}}))"), 1),
]
TOKEN = re.compile(r"[\w/]+|[^\w\s]", re.U)


def spans_entites(ligne):
    """Retourne la liste (debut, fin, entité) ; en cas de chevauchement, la première règle gagne."""
    spans = []
    for ent, rx, g in REGLES:
        for m in rx.finditer(ligne):
            d, f = m.span(g)
            if not any(d < f2 and f > d2 for d2, f2, _ in spans):
                spans.append((d, f, ent))
    return spans


def annoter_ligne(ligne):
    spans = spans_entites(ligne)
    sortie = []
    for m in TOKEN.finditer(ligne):
        tag = "O"
        for d, f, ent in spans:
            if m.start() >= d and m.end() <= f:
                debut = not any(d <= s.start() < m.start() and s.end() <= f for s in TOKEN.finditer(ligne))
                tag = ("B-" if debut else "I-") + ent
                break
        sortie.append((m.group(), tag))
    return sortie


def main():
    rnd = random.Random(42)
    choix = []
    for dossier in ("data/immobilier", "data/statut_personnel"):
        fichiers = sorted(f for f in glob.glob(os.path.join(dossier, "*.txt")) if os.path.getsize(f) > 0)
        rnd.shuffle(fichiers)
        choix += fichiers[:50]
    os.makedirs("annotations", exist_ok=True)
    cpt = Counter()
    with open("annotations/pre_annotation.conll", "w", encoding="utf-8") as out, \
         open("annotations/fichiers_choisis.txt", "w", encoding="utf-8") as lst:
        for chemin in choix:
            lst.write(chemin + "\n")
            texte = open(chemin, encoding="utf-8", errors="ignore").read()
            for ligne in texte.splitlines():
                if not ligne.strip():
                    continue
                for tok, tag in annoter_ligne(ligne):
                    out.write(f"{tok}\t{tag}\n")
                    if tag.startswith("B-"):
                        cpt[tag[2:]] += 1
                out.write("\n")
    print("Fichiers annotés :", len(choix))
    print("Entités trouvées (nombre d'occurrences) :")
    for e in ENTITES:
        print(f"  {e:13s} {cpt[e]}")


if __name__ == "__main__":
    main()
