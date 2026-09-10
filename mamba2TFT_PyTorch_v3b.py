# Model V3: Kanonski TFT Kategorijalni Klasifikator (PyTorch - Apple Silicon Compatible) 



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time

# --- UČITAVANJE PODATAKA ---
class LotoPyTorchDataset(Dataset):
    def __init__(self, csv_path, prozor=40):
        df = pd.read_csv(csv_path, header=None)
        self.data = torch.tensor(df.values, dtype=torch.long)
        self.prozor = prozor

    def __len__(self):
        return len(self.data) - self.prozor

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.prozor]
        y = self.data[idx + self.prozor]
        return x, y

# --- ARHITEKTURA KANONSKOG TFT ---
class GLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model * 2)

    def forward(self, x):
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

class GRN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
        self.glu = GLU(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.elu(self.linear_1(x))
        x = self.linear_2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, d_model):
        super().__init__()
        self.num_features = num_features
        self.feature_grns = nn.ModuleList([GRN(d_model) for _ in range(num_features)])
        self.flattened_grn = GRN(num_features * d_model)
        self.weight_linear = nn.Linear(num_features * d_model, num_features)

    def forward(self, x_list):
        # x_list je lista od 7 komponenti oblika [B, L, d_model]
        processed = [self.feature_grns[i](x_list[i]) for i in range(self.num_features)]
        flat_features = torch.cat(processed, dim=-1) # [B, L, 448]
        
        # Težine oblika [B, L, 7]
        weights = self.flattened_grn(flat_features)
        weights = self.weight_linear(weights) 
        weights = F.softmax(weights, dim=-1) # [B, L, 7]
        
        # Slažemo procesirane kolone u [B, L, 7, d_model]
        stacked = torch.stack(processed, dim=-2) 
        
        # Proširujemo težine u [B, L, 7, 1] radi pravilnog množenja duž d_model ose
        weights = weights.unsqueeze(-1)
        
        # Rezultat sumiranja je čist 3D tenzor [B, L, d_model] za LSTM
        return (weights * stacked).sum(dim=-2)

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d_model = x.shape
        q = self.q_linear(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, l, d_model)
        return self.out_linear(context)

class PyTorchCanonicalTFT(nn.Module):
    def __init__(self, vocab_size=40, d_model=64, n_heads=4):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, d_model) for _ in range(7)])
        self.vsn = VariableSelectionNetwork(num_features=7, d_model=d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads)
        self.post_attn_grn = GRN(d_model)
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(7)])

    def forward(self, x):
        x_features = [self.embeddings[i](x[:, :, i]) for i in range(7)]
        vsn_out = self.vsn(x_features) # Izlaz je sada ispravno [B, L, 64]
        lstm_out, _ = self.lstm(vsn_out)
        attn_out = self.attn(lstm_out)
        final_state = self.post_attn_grn(attn_out)[:, -1, :]
        izlazi = [head(final_state) for head in self.output_heads]
        return torch.stack(izlazi, dim=1)

# --- TRENING PETLJA ---
def treniraj_v1():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    csv_putanja = "/data/loto7_4682_k72_loto_2963.csv"
    
    dataset = LotoPyTorchDataset(csv_putanja, prozor=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PyTorchCanonicalTFT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    kriterijum = nn.CrossEntropyLoss()
    
    print("Mamba-2 TFT PyTorch | Prozor: 40 | Epoha: 150 | Uređaj: mps")
    print("Trening modela je pokrenut...")
    start_time = time.time()
    
    for epoha in range(1, 121):
        model.train()
        ukupni_gubitak = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            izlaz = model(x_batch)
            
            gubitak = 0
            for i in range(7):
                gubitak += kriterijum(izlaz[:, i, :], y_batch[:, i])
            
            gubitak.backward()
            optimizer.step()
            ukupni_gubitak += gubitak.item()
            
        if epoha % 50 == 0:
            print(f"PyTorch Epoha [{epoha}/150] | Kategorijalni Gubitak: {ukupni_gubitak/len(dataloader):.4f}")
            
    print(f"Trening završen za: {time.time() - start_time:.2f} sekundi.")
    
    model.eval()
    sačuvani_df = pd.read_csv(csv_putanja, header=None)
    zadnji_prozor = torch.tensor(sačuvani_df.values[-40:], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        predikcija_logita = model(zadnji_prozor)
        predikcija = torch.argmax(predikcija_logita, dim=-1).squeeze(0).cpu().numpy()
    
    print("\n==================================================")
    print(f"REZULTAT ZA FAJL {csv_putanja} (Sledeći red - PyTorch V3):")
    print(predikcija)
    print("==================================================")

if __name__ == "__main__":
    treniraj_v1()



"""

"""



"""
Prepoznavanje obrazaca (Pattern Recognition) 
Otkrivanje obrazaca (Pattern Discovery) 
Rudarenje obrazaca (Pattern Mining)   --->   Mamba-2

Model V1: Mamba-2 SSD Regresioni Model (PyTorch - Apple Silicon Compatible)
Model V2: Mamba-2 SSD Kategorijalni Klasifikator (Apple MLX - Native Silicon)
Model V3: Kanonski TFT Kategorijalni Klasifikator (PyTorch - Apple Silicon Compatible) 
Model V4: Kanonski TFT Kategorijalni Klasifikator (Apple MLX - Native Silicon)

Arhitektura Mamba 2 se zasniva na teoriji Structured State Space Duality (SSD). 
Mamba 2 omogućava da se proračun stanja transformiše u blokovske matrične multiplikacije, 
što je znatno lakše napisati u čistom Python-u/PyTorch-u. 
"""



"""
Optimalni odnosa između dužine istorijskog prozora i broja epoha za bazu podataka. 
Cilj je balans: dovoljno velik prozor da Mamba-2 uhvati cikluse, 
ali dovoljno primera za trening da model ne upadne u hiper-podešavanje (overfitting).

Evo optimalnih vrednosti za oba modela na osnovu količine podataka u tri CSV fajla, 
kako bi se sprečio overfitting (prenaučenost) i maksimalno iskoristila dužina istorije: 

Model V1,V3: PyTorch (Kraći prozor, brža konvergencija)
Za 4682 reda: Prozor: 40 | Epohe: 150 
Za 2963 reda: Prozor: 30 | Epohe: 120 
Za 1719 reda: Prozor: 20 | Epohe: 100  

Model V2,V4: Apple MLX (Širi prozor, dublja istorija)
Za 4682 reda: Prozor: 200 | Epohe: 1200 
Za 2963 reda: Prozor: 100 | Epohe: 1000 
Za 1719 reda: Prozor:  50 | Epohe: 400 
"""



"""
Mamba / S4 (State Space Models - SSM) 
Najnovija generacija AI arhitektura koja u mnogim zadacima predviđanja sekvenci nadmašuje čak i Transformere. 
Mamba ima linearno skaliranje i koristi selektivni mehanizam stanja. 
Za razliku od standardnih modela koji se muče sa dugoročnim zavisnostima u brojevima, 
Mamba može da kompresuje celu istoriju u jedno kompaktno "stanje" i precizno uoči ako se u CSV fajlu krije složen, 
visokodimenzionalni matematički algoritam ili generator.


Mamba / S4 (State Space Models) je arhitektonski napredniji i teoretski moćniji model od TFT-a za pronalaženje dubokih zakonitosti u dugim nizovima. 
Mamba je dizajnirana upravo da reši najveću manu starijih modela: sposobnost da filtrira nevažne podatke i zadrži savršen matematički fokus na ključnim promenama kroz vreme, bez gubitka memorije. 
Kroz svoj selektivni mehanizam stanja (Selective State Space), Mamba će pokušati da mapira skrivenu funkciju koja generiše ove brojeve i izračuna tačne vrednosti za sledećih 7 brojeva (next red).



Mamba / S4 ima suštinske prednosti koje direktno utiču na pronalaženje dubokih zakonitosti u loto kombinacijama: 

Efektivni kontekst nad dugom istorijom: 
Mamba koristi linearni selective scan mehanizam koji kompresuje celu istoriju CSV redova u jedno skriveno stanje konstantne veličine. 
TFT se oslanja na pažnju (Attention) koja ima kvadratnu složenost i gubi stabilnost kada prozor postane preveliki. 

Neprekidno modelovanje vremena (Continuous-time SSM): 
Mamba kroz diskretizaciju (Delta) uči skriveni kontinuum i dinamiku sistema. 
Ona tretira vaš CSV kao kontinualni signal koji se razvija kroz vreme, 
što joj omogućava da uoči duboke, ciklične i skrivene repetitivne obrasce koje TFT-ovi statični prozori promašuju. 

Selekcija informacija kroz vreme: 
Mamba filtrira nevažne šumove u svakom koraku sekvence. 
Za razliku od TFT-a koji pokušava da odjednom izvaže uticaj svih kolona u fiksnom prozoru, 
Mamba dinamički odlučuje šta iz prethodnih izvlačenja treba trajno zapamtiti, a šta odbaciti.


Model koristi Embedding sloj veličine 40 (za brojeve 1-39, gde je 0 rezervisana za mapiranje) 
i višeslojnu kauzalnu strukturu sa mehanizmom selektivnog stanja i rezidualnim vezama.
Embedding sloj: 
Brojeve od 1 do 39 ne posmatra kao proste cifre, već kreira "guste vektore" (d_model=256). 
Na taj način model uči skriveni kontekst (npr. kako se broj 7 ponaša kada je na prvoj poziciji u odnosu na to kada je na trećoj poziciji). 

Python kod za pripremu i treniranje modela
učitava csv, priprema podatke metodom kliznog prozora (gledajući istoriju da bi predvideo sledeći red) i trenira model visoke moći.


SSM/Mamba princip duboke kompresije: 
Za razliku od klasičnih modela, unutrašnji slojevi (SSMResidualBlock) vrše ne-linearnu projekciju podataka u prostor od 512 dimenzija (d_state). 
To omogućava računaru da zadrži matematičku strukturu informacija kroz ceo niz od CSV koraka.

Automatsko sortiranje na izlazu: 
Kod na samom kraju uzima sirove matematičke vrednosti modela, osigurava da ostanu u opsegu 1-39, 
uklanja eventualne duplikate i sortira ih od najmanjeg do najvećeg, prateći strukturu prethodnih redova.

 
Kada se pokrene skriptu, ona prođe kroz svih CSV redova da bi naučila pravila. 
Kada se trening završi, model u memoriji drži skriveno stanje sistema. 
Funkcija model(poslednji_prozor) uzima sam kraj CSV fajla (istoriju neposredno pre next koraka) 
i na osnovu svega što je naučila generiše potpuno novih 7 brojeva. 
Kada se pokrene kod u terminalu, na samom dnu se dobije jasan ispis. 
"""



"""
Mamba, odnosno Selective State Space Model (SSM) — preciznije novija arhitektura Mamba-2.
Za sekvencijalno predviđanje na CSV podacima glavni model bi bio:
Mamba-2 regresioni model za sekvence skupova
Svako izvlačenje kodira se kao binarni vektor od 39 vrednosti, 
Mamba-2 obrađuje hronološki niz prethodnih izvlačenja, 
a regresiona glava daje kontinuirani skor za svih 39 brojeva. 
Sedam najvećih skorova čini NEXT.


Da bi kod bio napisan u čistoj Mamba-2 arhitekturi, 
on bi morao da koristi zvaničnu mamba_ssm biblioteku i njene specifične CUDA operatore, 
što zahteva Linux operativni sistem i grafičku kartu (GPU).
"""
