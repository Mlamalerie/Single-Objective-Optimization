# -*- coding: Latin-1 -*-
# programme de resolution du probleme du voyaguer de commerce
# par l'algorithme du recuit simule
# Dominique Lefebvre pour TangenteX.com
# Peio Loubiere pour l'EISTI
# septembre 2017
# usage : python RSTSP.py NOMFICHIER
from scipy import *
from math import *
from matplotlib.pyplot import *
from functools import *
import sys
import random as rnd

# Instance du problème
#FIC="NUMERO.tsp"
if (len(sys.argv) > 1):
    FIC=sys.argv[1]
else:
    print("Aucun fichier spécifié...")
    sys.exit("USAGE : python RSTSP.py NUMERO_INSTANCE.tsp")

# ###################################### Parametres de PSO ##################################

# params facteur de contrition
ksi, c1, c2 = 0.7298844, 2.05, 2.05
# params usuels
psi,cmax = (0.7, 1.47) 
# psi,cmax = (0.8, 1.62)
# psi,cmax = (1, 1) 
# ##############################################################################################

# Creation de la figure
TPSPause = 0.1 # pour affichage
fig1 = figure()
canv = fig1.add_subplot(1,1,1)
xticks([])
yticks([])

# Parsing du fichier de données
# pre-condition : nomfic : nom de fichier (doit exister)
# post-condition : (x,y) coordonnees des villes
def parse(nomfic):
    absc=[]
    ordo=[]
    with open(nomfic,'r') as inf:
        for line in inf:
            absc.append(float(line.split(' ')[1]))
            ordo.append(float(line.split(' ')[2]))
    return (array(absc,dtype=float),array(ordo,dtype=float))


# Affiche les coordonnées des points du chemin ainsi que le meilleur trajet trouvé et sa longueur
# pre-conditions :
#   - best_trajet, best_dist : meilleur trajet trouvé et sa longueur,
def affRes(best):
    print("trajet = {}".format(best['bestpos']))
    print("distance = {}".format(best['bestfit']))

# Rafraichit la figure du trajet, on trace le meilleur trajet trouvé
# pre-conditions :
#   - best_trajet, best_dist : meilleur trajet trouvé et sa longueur,
#   - x, y : tableaux de coordonnées des points du chemin
def dessine(best_trajet, best_dist, x, y):
    global canv,lx,ly
    canv.clear()
    canv.plot(x[best_trajet],y[best_trajet],'k')
    canv.plot([x[best_trajet[-1]], x[best_trajet[0]]],[y[best_trajet[-1]], \
      y[best_trajet[0]]],'k')
    canv.plot(x,y,'ro')
    title("Distance : {}".format(best_dist))
    pause(TPSPause)


# Figure des graphes de :
#   - l'ensemble des energies des fluctuations retenues
#   - la meilleure energie
#   - la decroissance de temperature
def dessineStats(Htemps, Hbest):
    # affichage des courbes d'evolution
    fig2 = figure(2)
    subplot(1,1,1)
    semilogy(Htemps, Hbest)
    title('Evolution de la meilleure distance')
    xlabel('Temps')
    ylabel('Distance')
    show()

# Factorisation des fonctions de calcul de la distance totale
# pre-condition : p1(x1,y1),p2(x2,y2) coordonnees des villes
# post-condition : distance euclidienne entre 2 villes
def distance(p1,p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Fonction de calcul de lla longueur du trajet,
# pre-condition : 
#   - coords : coordonnées des points du trajet
#   - trajet : ordre de parcours des villes
# post-condition : Pb du VC : la distance totale du trajet
def eval(coords,trajet,dim):
    longueur = 0.0
    coord = coords[trajet]
    for i in range(-1,dim-1): # on calcule la distance en fermant la boucle
        longueur += distance(coord[i], coord[i+1])
    return longueur

# Crée une particule 
# une particule est décrite par : 
#   - pos : solution liste des variables
#   - vit : vitesse de déplacement (nulle à l'initialisation)
#   - fit :  aire du rectangle
#   - bestpos : meilleure configuration visitée
#   - bestfit : évaluation de la meilleure configuration visitée
#   - bestvois : meilleur voisin (global pour cette version)
def initUn(dim,coords):
    pos = rnd.sample(range(dim),dim)
    fit = eval(coords,pos,dim)
    return {'vit':[], 'pos':pos, 'fit':fit, 'bestpos':pos, 'bestfit':fit, 'bestvois':[]}

# Init de la population
def initEssaim(nb,dim,coords):
    return [initUn(dim,coords) for i in range(nb)]

# Retourne la particule de meilleure fitness
def maxPartic(p1,p2):
    if (p1["fit"] < p2["fit"]):
        return p1 
    else:
        return p2

# Retourne une copie de la particule de meilleure fitness dans la population
def getBest(essaim):
    return dict(reduce(lambda acc, e: maxPartic(acc,e),essaim[1:],essaim[0]))

# Mise à jour des infos pour les paticules de la population
def maj(partic,bestpart):
    nv = dict(partic)
    if(partic["fit"] < partic["bestfit"]):
        nv['bestpos'] = partic["pos"][:]
        nv['bestfit'] = partic["fit"]
    nv['bestvois'] = bestpart["bestpos"][:]
    return nv

def majlocal(partic,essaim,nb,nbvois):
    i = essaim.index(partic)
    l = [essaim[(i+j)%nb] for j in range(1,nbvois+1)]
    bestpart = getBest(l)
    nv = dict(partic)
    if(partic["fit"] < partic["bestfit"]):
        nv['bestpos'] = partic["pos"][:]
        nv['bestfit'] = partic["fit"]
    nv['bestvois'] = bestpart["bestpos"][:]
    return nv


def moins(pos1, pos2):
    d = len(pos1)
    vitres = []
    poscopie = pos1[:]
    for i in range(d):
        e = pos2[i]
        j = poscopie.index(e)
        if (i!=j):
            vitres.append((i,j))
            poscopie[i], poscopie[j] = poscopie[j], poscopie[i]

    return vitres

def plus(vit1, vit2):
    vitres = vit1 + vit2
    return vitres

def fois(k, vit):
    vitres = []
    while(k>=1):
        # vitres = vitres + vit
        k -= 1
    tronc = int(round(k*len(vit)))
    for i in range(tronc):
        vitres.append(vit[i])
    return vitres

def move(part, vit):
    partres = part[:]
    for (i,j) in vit:
        partres[i], partres[j] = partres[j], partres[i]        
    return partres

# Calcule la vitesse et déplace une paticule
def deplace(partic,dim,coords):
    global ksi,c1,c2,psi,cmax
    
    nv = dict(partic)

    # vitesse = plus(fois(psi,partic["vit"]), \
    #                 plus(fois(cmax*random.uniform(),moins(partic["bestpos"], partic["pos"])), \
    #                     fois(cmax*random.uniform(),moins(partic["bestvois"],partic["pos"]))))
    vitesse = plus(fois(ksi,partic["vit"]),plus(fois(c1*random.uniform(),moins(partic["bestpos"], partic["pos"])),fois(c2*random.uniform(),moins(partic["bestvois"],partic["pos"]))))
    position = move(partic['pos'], vitesse)    

    nv['vit'] = vitesse
    nv['pos'] = position
    nv['fit'] = eval(coords,position,dim)

    return nv

# ##################################### INITIALISATION DE L'ALGORITHME ############################
# Construction des données depuis le fichier
(x,y) = parse(FIC) # x,y sont gardés en l'état pour l'affichage graphique
coords = array(list(zip(x,y)),dtype=float) # On contruit le tableau de coordonées (x,y)



# Paramètre du probleme
dim = len(x)    # nombre de villes
Nb_cycles = 1000*dim
Nb_partic = 10+2*int(ceil(sqrt(dim)))
Nb_vois = 3
Htemps = []       # temps
Hbest = []        # distance

# ##################################### BOUCLE PRINCIPALE DE L'ALGORITHME ############################

# initialisation de la population
essaim = initEssaim(Nb_partic,dim,coords)
# initialisation de la meilleure solution
best = getBest(essaim)
best_cycle = best

# on trace le chemin de depart
dessine(best['bestpos'], best['bestfit'], x, y)

for i in range(Nb_cycles):
    # Mise à jour des informations
    # essaim = [maj(e,best_cycle) for e in essaim]
    essaim = [majlocal(e,essaim,Nb_partic,Nb_vois) for e in essaim]
    # calculs de vitesse et déplacement
    essaim = [deplace(e,dim,coords) for e in essaim]
    # Mise à jour de la meilleure solution
    best_cycle = getBest(essaim)
    if (best_cycle["bestfit"] < best["bestfit"]):
        best = best_cycle
        dessine(best['bestpos'], best['bestfit'], x, y)

    # historisation des donnees
    if i % 10 == 0:
        Htemps.append(i)
        Hbest.append(best['bestfit'])


# ##################################### FIN DE L'ALGORITHME - AFF DES RÉsSULTATS ############################
# affichage console du résultat
affRes(best)
# graphique des stats
dessineStats(Htemps, Hbest)
