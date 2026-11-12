"""Prithvi queda congelado; esta red pequeña aprende el pronóstico espacial y temporal."""
import torch
from torch import nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(channels+hidden, 4*hidden, 3, padding=1)

    def forward(self, x, state):
        if state is None:
            h = x.new_zeros(x.shape[0],self.hidden,*x.shape[-2:])
            cell = torch.zeros_like(h)
        else:
            h,cell = state
        i,f,o,g = self.gates(torch.cat([x,h],1)).chunk(4,1)
        cell = torch.sigmoid(f)*cell + torch.sigmoid(i)*torch.tanh(g)
        h = torch.sigmoid(o)*torch.tanh(cell)
        return h,cell


class ForecastHead(nn.Module):
    def __init__(self, feature_channels, patch_size, hidden, projected, mean, std):
        super().__init__()
        pt,ph,pw = patch_size
        if pt != 1 or ph != pw:
            raise ValueError("Se requieren parches espaciales cuadrados de una fecha.")
        self.patch = ph
        self.project = nn.Conv2d(feature_channels, projected, 1)
        # Tres canales conocidos de fecha/horizonte + fracción válida por parche.
        self.temporal = ConvLSTMCell(projected+4, hidden)
        self.decode = nn.Conv2d(hidden, 6*ph*pw, 1)
        # Al iniciar reproduce la referencia de persistencia; después aprende sus cambios.
        nn.init.zeros_(self.decode.weight)
        nn.init.zeros_(self.decode.bias)
        self.register_buffer("lower_bound", (-torch.tensor(mean)/torch.tensor(std)).reshape(1,6,1,1))

    def forward(self, batch):
        features = batch["features"] # B,T,C,16,16 para el 600M a 224 px
        state = None
        for t in range(features.shape[1]):
            x = F.gelu(self.project(features[:,t]))
            known = batch["known_time"][:,t,:,None,None].expand(-1,-1,*x.shape[-2:])
            x = torch.cat([x,known,batch["quality"][:,t]],1)
            state = self.temporal(x,state)
        delta = F.pixel_shuffle(self.decode(state[0]), self.patch)
        # La reflectancia armonizada es no negativa; el límite se aplica también al evaluar.
        return torch.maximum(batch["baseline"]+delta, self.lower_bound)


def make_head(c, meta):
    f = c["forecast"]
    return ForecastHead(meta["embedding_dim"], meta["patch_size"], f["hidden_channels"],
                        f["project_channels"], meta["mean"], meta["std"])


def forecast_loss(pred, batch):
    valid = batch["valid"].unsqueeze(1).expand_as(pred)
    den = valid.sum()
    if den.item() == 0:
        raise ValueError("No hay píxeles comunes válidos entre entrada y objetivo.")
    num = ((pred.float()-batch["target"].float()).square()*valid).sum()
    return num,den
