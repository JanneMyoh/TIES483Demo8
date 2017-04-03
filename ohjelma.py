# -*- coding: cp1252 -*-
import numpy as np
import random
from scipy.optimize import fmin_cg
from multiprocessing import Process, Pipe
import time

#LJ ryhmän atomien luku määrä
nAtoms = 6

#Laskee potentiaali energian annetuilla paikoilla. P on 3*n pituinen vektori missä atomien koordinaatit ovat [x_0 y_0 z_0 x_1 ... x_n y_n z_n]
def summaFunk(p):
    #summaa eri etäisyydet kun i < j
    summa = 0;
    for i in range(0,nAtoms):
        for j in range(i+1,nAtoms):
            if( i != j):
                rij = ((p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2 + (p[i*3+2]-p[j*3+2])**2)
               # print str(i) + " " + str(j) + " " + str(rij)
                insi = (1/rij)**12 - (1/rij)**6
                summa = summa + insi
    return 4*summa

#savattaa tai vähentää jokaista koordinaattia enintään 0.2
def moveAtoms(p):
    pal = []
    for i in p:
        pal.append(i + (random.random()*0.4 - 0.2))
    return pal

#Suorittaa kulma liikkeen annetulle koordinaatille
#Tar: muutettavat koordinaatit karteesisessa koordinaatistossa
#mat: [r, theta , phi]. r asetetaan suoraan, kun theta ja phi lisätään atomin koordinaattiin
def angularMove(tar,amt):
    #muutetaan pallo koordinaatistoon
    xy = tar[0]**2 + tar[1]**2
    r = np.sqrt(xy + tar[2]**2)
    hor = np.arccos(tar[2]/r)
    ver = np.arctan(tar[1]/tar[0])
    #tehdään muutokset
    r = amt[0]
    hor = hor + amt[1]
    ver = ver + amt[2]
    #palautetaan karteesiseen koordinaatistoon
    x = r*np.sin(hor)*np.cos(ver)
    y = r*np.sin(hor)*np.sin(ver)
    z = r*np.cos(hor)
    return [x,y,z]

#palauttaa indeksin jossa atomi jota liikautetaan jos liikautusta ei tule tehdä palautetaan -1
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
    #ollaan laskettu pari energiat. Tutkitaan tarvitseeko tehdä liikutusta
    mxE = np.amax(pEner)
    mnE = np.amin(pEner)
    if(np.fabs(mxE/mnE) >= 1.5):
        #Atomia tulee liikuttaa
        return pEner.index(mxE)
    else:
        #Atomia ei tarvitse liikuttaa
        return -1

#Työläis prosessin suorittama metodi
#pos: koordinaatit lähtötilanteessa
#conn: putki jota pitkin työläinen ja pää prosessi kommunikoivat
def kasittelija(pos,conn):
    #Suoritetaan silmukkaa kunnes pää ohjelma käskee lopetuksen. Ei siis luoda jokaiselle minimoinnille omaa prosessia
    while pos != -1:
        pos = moveAtoms(pos)
        #tarkistetaan tuleeko tehdä kulma liikettä
        #muutetaan vektori matriisiksi
        posM = [[pos[i*3],pos[i*3+1],pos[i*3+2]] for i in range(len(pos)/3)]
        #tarkistetaan tuleeko jotain atomia kulma liikuttaa
        liikA = checkPairEnergy(posM)
        if liikA != -1:
            #atomia liikA tulee liikuttaa. Toteutetaan se. Lasketaan aluksi massa keskipiste ja atomien etäisyys siitä
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
        #etsitään rakenteen potentiaali energia    
        res = summaFunk(fmin_cg(summaFunk,np.array(pos),gtol=0.01, disp=0))
        #lähetetään tulokset pää prosessille
        conn.send(res)
        conn.send(pos)
        #odotetaan pää prosessilta seuraavaa rakennetta
        pos = conn.recv()

#Osa jonka vain pää prosessi suorittaa
if __name__ == '__main__':
    #Suoritettavien iteraatioiden määrä
    nIter = 3000
    #Työläis prosessien lkm.
    nThreads = 3

    #luodaan alku asetelma ja lasketaan sen potentiaali energia
    pos = []
    for i in range(0,nAtoms*3):
        pos.append(random.random()*4)

    res = summaFunk(fmin_cg(summaFunk,np.array(pos),gtol=0.01,disp=0))

    #otetaan muistiin parhaat (tässä kohtaa ainoat) rakenteet ja sitä vastaava energia
    bestPos = pos
    bestE = res

    #Alustetaan muuttujia
    pipes = []
    threads = []
    vals = 0
    pstns = 0

    #luodaan kommunikaatio putket, ja työläis prosessit
    for i in range(nThreads):
        start,end = Pipe()
        pipes.append(start)
        threads.append(Process(target=kasittelija, args=(bestPos,end)))
    #Alkeellinen tapa selvittää operaation käyttämä aika
    tStart = time.ctime()
    #käynnistetään prosessit
    for i in threads:
        i.start()
    #Itse pää silmukka
    j = 0
    while j < nIter:
        #Käsitellään yksi prosessi kerrallaan. Otetaan vastaan sen tulokset, verrataan sitä nykyiseen parhaaseen ja tehdään päätös pidetäänkö se vai vanha
        for i in range(nThreads):
            vals = pipes[i].recv()
            pstns = pipes[i].recv()
            if vals <= bestE:
                bestE = vals
                bestPos = pstns
            j = j+1
            print j
            #Tulostetaan välillä paras löydetty energia
            if(j%100 == 0):
                print bestE
        #Lähetetään työläisille seuraava lähtö rakenne
        for p in pipes:
            p.send(bestPos)
    #Ollaan suoritettu haluttu määr iteraatioita. Suljetaan työläis prosessit
    for p in pipes:
        p.send(-1)
    tEnd = time.ctime()
    #Kirjataan tulokset tiedostoon ja tulostetaan käyttäjälle
    print "started " + tStart
    print "ended " + tEnd
    #Lopullinen rakenne tulee vielä selvittää, sillä bestPos on vain eräs rakenne jonka minimoimalla päästään optimi rakenteeseen
    bestPos = fmin_cg(summaFunk,np.array(pos),gtol=0.0001,disp=0)
    f = open("finalPos.txt","w")
    f.write(str(bestPos))
    f.write(str(bestE))
    f.close()
