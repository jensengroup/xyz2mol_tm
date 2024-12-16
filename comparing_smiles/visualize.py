import py3Dmol
from rdkit import Chem


def draw_mol(
    mol,
    width=800,
    height=800,
    Hs=False,
    confId=-1,
    multipleConfs=False,
    atomlabel=False,
    hit_ats=None,
    gen_struct=None,
    trajectory=False,
):
    p = py3Dmol.view(width=width, height=height)

    if isinstance(mol, str):
        xyz_f = open(mol)
        line = xyz_f.read()
        xyz_f.close()
        p.addModel(line, "xyz")
    else:
        if multipleConfs:
            for conf in mol.GetConformers():
                mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                p.addModel(mb, "sdf")
        else:
            mb = Chem.MolToMolBlock(mol, confId=confId)
            p.addModel(mb, "sdf")

    p.setStyle({"stick": {"radius": 0.17}, "sphere": {"radius": 0.4}})
    p.setStyle({"elem": "H"}, {"stick": {"radius": 0.17}, "sphere": {"radius": 0.28}})
    if atomlabel:
        p.addPropertyLabels("index")  # ,{'elem':'H'}
    p.setClickable(
        {},
        True,
        """function(atom,viewer,event,container) {
        if(!atom.label) {
            atom.label = viewer.addLabel(atom.index,{position: atom, backgroundColor: ‘white’, fontColor:‘black’});
        }}""",
    )

    p.zoomTo()
    p.update()
