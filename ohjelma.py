# -*- coding: cp1252 -*-
import numpy as np
import random
from scipy.optimize import fmin_cg
from multiprocessing import Process, Pipe
import time

#LJ ryhm�n atomien luku m��r�
nAtoms = 6

#Laskee potentiaali energian annetuilla paikoilla. P on 3*n pituinen vektori miss� atomien koordinaatit ovat [x_0 y_0 z_0 x_1 ... x_n y_n z_n]
def summaFunk(p):
    #summaa eri et�isyydet kun i < j
    summa = 0;
    for i in range(0,nAtoms):
        for j in range(i+1,nAtoms):
            if( i != j):
                rij = ((p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2 + (p[i*3+2]-p[j*3+2])**2)
               # print str(i) + " " + str(j) + " " + str(rij)
                insi = (1/rij)**12 - (1/rij)**6
                summa = summa + insi
    return 4*summa

#savattaa tai v�hent�� jokaista koordinaattia enint��n 0.2
def moveAtoms(p):
    pal = []
    for i in p:
        pal.append(i + (random.random()*0.4 - 0.2))
    return pal

#Suorittaa kulma liikkeen annetulle koordinaatille
#Tar: muutettavat koordinaatit karteesisessa koordinaatistossa
#mat: [r, theta , phi]. r asetetaan suoraan, kun theta ja phi lis�t��n atomin koordinaattiin
def angularMove(tar,amt):
    #muutetaan pallo koordinaatistoon
    xy = tar[0]**2 + tar[1]**2
    r = np.sqrt(xy + tar[2]**2)
    hor = np.arccos(tar[2]/r)
    ver = np.arctan(tar[1]/tar[0])
    #tehd��n muutokset
    r = amt[0]
    hor = hor + amt[1]
    ver = ver + amt[2]
    #palautetaan karteesiseen koordinaatistoon
    x = r*np.sin(hor)*np.cos(ver)
    y = r*np.sin(hor)*np.sin(ver)
    z = r*np.cos(hor)
    return [x,y,z]

#palauttaa indeksin jossa atomi jota liikautetaan jos liikautusta ei tule tehd� palautetaan -1
def checkPairEnergy(pos):

    pEner = []
    for n in range(len(pos)):
        summa = 0
        #Lasketaan pari energia atomille
        for i in range(len(pos)):
            if i != n:
                r = np.sqrt((pos[i][0] - pos[n][0])**2 + (pos[i][1] - pos[n][1])**2 + (pos[i][2] - pos[n][2])**2)
                summa = summa + ((1/r)**12 - (1/r)**6)
        pEner.append(summa*4)
    #ollaan laskettu pari energiat. Tutkitaan tarvitseeko tehd� liikutusta
    mxE = np.amax(pEner)
    mnE = np.amin(pEner)
    if(np.fabs(mxE/mnE) >= 1.5):
        #Atomia tulee liikuttaa
        return pEner.index(mxE)
    else:
        #Atomia ei tarvitse liikuttaa
        return -1

#Ty�l�is prosessin suorittama metodi
#pos: koordinaatit l�ht�tilanteessa
#conn: putki jota pitkin ty�l�inen ja p�� prosessi kommunikoivat
def kasittelija(pos,conn):
    #Suoritetaan silmukkaa kunnes p�� ohjelma k�skee lopetuksen. Ei siis luoda jokaiselle minimoinnille omaa prosessia
    while pos != -1:
        pos = moveAtoms(pos)
        #tarkistetaan tuleeko tehd� kulma liikett�
        #muutetaan vektori matriisiksi
        posM = [[pos[i*3],pos[i*3+1],pos[i*3+2]] for i in range(len(pos)/3)]
        #tarkistetaan tuleeko jotain atomia kulma liikuttaa
        liikA = checkPairEnergy(posM)
        if liikA != -1:
            #atomia liikA tulee liikuttaa. Toteutetaan se. Lasketaan aluksi massa keskipiste ja atomien et�isyys siit�
            avg = np.average(posM,axis=0)
            dis = []
            for j in range(len(posM)):
                spo = [posM[j][i]-avg[i] for i in range(0,3)]
                dis.append(np.sqrt(spo[0]**2 + spo[1]**2 + spo[2]**2))
            r = np.amax(dis)
            #Asetetaan MKP origoksi
            spo = [posM[liikA][i]-avg[i] for i in range(0,3)]
            #suoritetaan liikutus
            spoLi = angularMove(spo,[r,random.random()*2*np.pi , random.random()*np.pi])
            #palautetaan origo
            spoFi = [spoLi[i]+avg[i] for i in range(0,3)]
            #Laitetaan liikutettu atomi takaisin matriisiin
            posM[liikA] = spoFi
        #muutetaan matriisi vektoriksi
        pos = []
        for atom in posM:
            for cord in atom:
                pos.append(cord)
        #etsit��n rakenteen potentiaali energia    
        res = summaFunk(fmin_cg(summaFunk,np.array(pos),gtol=0.01, disp=0))
        #l�hetet��n tulokset p�� prosessille
        conn.send(res)
        conn.send(pos)
        #odotetaan p�� prosessilta seuraavaa rakennetta
        pos = conn.recv()

#Osa jonka vain p�� prosessi suorittaa
if __name__ == '__main__':
    #Suoritettavien iteraatioiden m��r�
    nIter = 3000
    #Ty�l�is prosessien lkm.
    nThreads = 3

    #luodaan alku asetelma ja lasketaan sen potentiaali energia
    pos = []
    for i in range(0,nAtoms*3):
        pos.append(random.random()*4)

    res = summaFunk(fmin_cg(summaFunk,np.array(pos),gtol=0.01,disp=0))

    #otetaan muistiin parhaat (t�ss� kohtaa ainoat) rakenteet ja sit� vastaava energia
    bestPos = pos
    bestE = res

    #Alustetaan muuttujia
    pipes = []
    threads = []
    vals = 0
    pstns = 0

    #luodaan kommunikaatio putket, ja ty�l�is prosessit
    for i in range(nThreads):
        start,end = Pipe()
        pipes.append(start)
        threads.append(Process(target=kasittelija, args=(bestPos,end)))
    #Alkeellinen tapa selvitt�� operaation k�ytt�m� aika
    tStart = time.ctime()
    #k�ynnistet��n prosessit
    for i in threads:
        i.start()
    #Itse p�� silmukka
    j = 0
    while j < nIter:
        #K�sitell��n yksi prosessi kerrallaan. Otetaan vastaan sen tulokset, verrataan sit� nykyiseen parhaaseen ja tehd��n p��t�s pidet��nk� se vai vanha
        for i in range(nThreads):
            vals = pipes[i].recv()
            pstns = pipes[i].recv()
            if vals <= bestE:
                bestE = vals
                bestPos = pstns
            j = j+1
            print j
            #Tulostetaan v�lill� paras l�ydetty energia
            if(j%100 == 0):
                print bestE
        #L�hetet��n ty�l�isille seuraava l�ht� rakenne
        for p in pipes:
            p.send(bestPos)
    #Ollaan suoritettu haluttu m��r iteraatioita. Suljetaan ty�l�is prosessit
    for p in pipes:
        p.send(-1)
    tEnd = time.ctime()
    #Kirjataan tulokset tiedostoon ja tulostetaan k�ytt�j�lle
    print "started " + tStart
    print "ended " + tEnd
    #Lopullinen rakenne tulee viel� selvitt��, sill� bestPos on vain er�s rakenne jonka minimoimalla p��st��n optimi rakenteeseen
    bestPos = fmin_cg(summaFunk,np.array(pos),gtol=0.0001,disp=0)
    f = open("finalPos.txt","w")
    f.write(str(bestPos))
    f.write(str(bestE))
    f.close()
