# -*- coding: utf-8 -*-
"""Fonctions de prétraitement partagées (nécessaires pour recharger les modèles joblib)."""
import re

TASHKEEL = re.compile(r"[\u064B-\u0652\u0640]")


def normaliser(t):
    t = TASHKEEL.sub("", t)
    t = re.sub("[إأآا]", "ا", t)
    return t.replace("ى", "ي").replace("ة", "ه")


def prep_recherche(t):
    return re.sub(r"\d+", " ", normaliser(t))
