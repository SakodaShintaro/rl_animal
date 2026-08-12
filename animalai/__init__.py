'''
Marks this directory as a regular package.

Without it, `animalai` here is a namespace package, and a regular package of the
same name wins the import no matter what comes first on sys.path. The image also
carries animalai 5.0.1 in site-packages for the Animal-AI v4 binary (see
aai4_eval.py), which silently shadowed this v1 package and broke
evaluate_competition.py.
'''
