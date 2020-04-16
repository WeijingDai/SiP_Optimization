#%% 
# Import Abaqus library
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
import os
import numpy as np
import material
import time
import job
import random

# Basic setting
os.chdir(r"C:\SUSTech-Postdoc\SDIM-Projects\Warpage\Genetic_Algorithm")
mdb.saveAs(pathName='C:/SUSTech-Postdoc/SDIM-Projects/Warpage/Genetic_Algorithm/ga.cae')

# Geometry parameter
i = 5
h_die_rg = np.linspace(200, 320, i, endpoint=True)
j = 5
h_adhesive_rg = np.linspace(20, 40, j, endpoint=True)
k = 5
h_sub_rg = np.linspace(200, 300, k, endpoint=True)
l = 5
h_emc_rg = np.linspace(550, 950, l, endpoint=True)
l_ratio = 0.001
l_emc = 4.5
l_die = 2.
l_adhesive = 2.
l_sub = 4.5
m = 5
emc_cte_rg = np.linspace(8, 12, m, endpoint=True)
cte_ratio = 0.000001
variable_num = 4

def cte_temp(ct=np.array([])):
    aba_ct = ct.copy()
    temp_num = len(ct)
    i = 1
    e_temp = np.zeros(ct.shape[-1] - 1)
    while i < temp_num:
        e_new = ct[i, 0:-1] * (ct[i, -1] - ct[i - 1, -1])
        e_temp += e_new
        a_temp = e_temp / (ct[i, -1] - ct[0, -1])
        aba_ct[i, 0:-1] = a_temp
        i += 1
    return tuple(map(tuple, aba_ct[1:]))

def abaqus_model(variable_list = []):
    h_die_um = int(h_die_rg[2])
    h_adhesive_um = int(h_adhesive_rg[variable_list[0]])
    h_sub_um = int(h_sub_rg[variable_list[1]])
    h_emc_um = int(h_emc_rg[variable_list[2]])
    emc_cte = int(emc_cte_rg[variable_list[3]])

    model_name = 'ke_ch2_3d-mm-merge-h_die_' + str(h_die_um) + '-h_adhesive_' + str(h_adhesive_um) + \
                '-h_sub_' + str(h_sub_um) + '-h_emc_' + str(h_emc_um)
    mdb.Model(name=model_name)
    m = mdb.models[model_name]
    a = m.rootAssembly
    session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
    m.Material(name='emc', description='mm, tone, s, K')
    m.materials['emc'].Elastic(table=((27000.0, 0.3),))

    m.Material(name='sub', description='mm, tone, s, K')
    m.materials['sub'].Elastic(type=ENGINEERING_CONSTANTS, table=((16892.0, 16892.0, 7377.0, 0.11, 0.39, 0.39, 7609, 2654, 2654),))
    m.materials['sub'].Expansion(type=ORTHOTROPIC, zero=25.0, temperatureDependency=ON, table=cte_temp(np.array([[0., 0., 0., 25.], [17.38e-6, 17.38e-6, 27.07e-6, 178.26], [17.38e-6, 17.38e-6, 172.1e-6, 250]])))

    m.Material(name='die', description='mm, tone, s, K')
    m.materials['die'].Elastic(table=((161000.0, 0.21),))
    m.materials['die'].Expansion(type=ISOTROPIC, zero=.0, temperatureDependency=ON, table=cte_temp(np.array([[.0, .0], [2.6e-6, 25.], [3.2e-6, 170.]])))

    m.Material(name='adhesive', description='mm, tone, s, K')
    m.materials['adhesive'].Elastic(temperatureDependency=ON, table=((1870.0, 0.4, 25.), (1460.0, 0.4, 50.),))
    m.materials['adhesive'].Expansion(type=ISOTROPIC, zero=25.0, temperatureDependency=ON, table=cte_temp(np.array([[.0, 25.0], [50.e-6, 100.], [100e-6, 250.]])))

    # Section definition
    m.HomogeneousSolidSection(name='emc', material='emc', thickness=None)
    m.HomogeneousSolidSection(name='sub', material='sub', thickness=None)
    m.HomogeneousSolidSection(name='die', material='die', thickness=None)
    m.HomogeneousSolidSection(name='adhesive', material='adhesive', thickness=None)

    h_emc = h_emc_um * l_ratio
    h_die = h_die_um * l_ratio
    h_adhesive = h_adhesive_um * l_ratio
    h_sub = h_sub_um * l_ratio
    Sketch = m.ConstrainedSketch(name='Sketch', sheetSize=200.0)
    Sketch.rectangle(point1=(-l_emc, -l_emc), point2=(l_emc, l_emc))
    emcPart = m.Part(name='emc', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    emcPart.BaseSolidExtrude(sketch=Sketch, depth=h_emc)

    Sketch = m.ConstrainedSketch(name='Sketch', sheetSize=200.0)
    Sketch.rectangle(point1=(-l_sub, -l_sub), point2=(l_sub, l_sub))
    subPart = m.Part(name='sub', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    subPart.BaseSolidExtrude(sketch=Sketch, depth=h_sub)
    subPart.Set(faces=subPart.faces.findAt(((0, 0, 0),)), name='sub_lower')
    subPart.Set(faces=subPart.faces.findAt(((0, 0, h_sub),)), name='sub_upper')

    Sketch = m.ConstrainedSketch(name='Sketch', sheetSize=200.0)
    Sketch.rectangle(point1=(-l_die, -l_die), point2=(l_die, l_die))
    diePart = m.Part(name='die', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    diePart.BaseSolidExtrude(sketch=Sketch, depth=h_die)
    diePart.Set(faces=diePart.faces.findAt(((0, 0, 0),)), name='die_lower')
    diePart.Set(faces=diePart.faces.findAt(((0, 0, h_die),)), name='die_upper')
    diePart.Set(faces=diePart.faces.findAt(((l_die, 0, h_die/2),)), name='die_side')

    Sketch = m.ConstrainedSketch(name='Sketch', sheetSize=200.0)
    Sketch.rectangle(point1=(-l_adhesive, -l_adhesive), point2=(l_adhesive, l_adhesive))
    adhesivePart = m.Part(name='adhesive', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    adhesivePart.BaseSolidExtrude(sketch=Sketch, depth=h_adhesive)
    adhesivePart.Set(faces=adhesivePart.faces.findAt(((0, 0, 0),)), name='adhesive_lower')
    adhesivePart.Set(faces=adhesivePart.faces.findAt(((0, 0, h_adhesive),)), name='adhesive_upper')
    adhesivePart.Set(faces=adhesivePart.faces.findAt(((l_adhesive, 0, h_adhesive/2),)), name='adhesive_side')

    ## Use Assembly module to create part from subtraction
    adhesiveIns = a.Instance(name='adhesive', part=adhesivePart, dependent=OFF)
    dieIns = a.Instance(name='die', part=diePart, dependent=OFF)
    a.translate(instanceList=('die',), vector=(0.0, 0.0, h_adhesive))
    emcIns = a.Instance(name='emc', part=emcPart, dependent=OFF)
    a.InstanceFromBooleanCut(name='emc-1', instanceToBeCut=a.instances['emc'], cuttingInstances=(adhesiveIns, dieIns,), originalInstances=DELETE)
    del a.features['emc-1-1']
    del m.parts['emc']
    emcPart = m.parts['emc-1']
    emcPart.Set(faces=emcPart.faces.findAt(((0, 0, h_die+h_adhesive),)), name='emc_lower')
    emcPart.Set(faces=emcPart.faces.findAt(((0, 0, h_emc),)), name='emc_upper')
    emcPart.Set(faces=emcPart.faces.findAt(((l_die, 0, h_die/2),)), name='emc_side')

    # Section assignment
    emcPart.Set(cells=emcPart.cells[0:], name='emc')
    emcPart.SectionAssignment(region=emcPart.sets['emc'], sectionName='emc', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    subPart.Set(cells=subPart.cells[0:], name='sub')
    subPart.SectionAssignment(region=subPart.sets['sub'], sectionName='sub', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    diePart.Set(cells=diePart.cells[0:], name='die')
    diePart.SectionAssignment(region=diePart.sets['die'], sectionName='die', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    adhesivePart.Set(cells=adhesivePart.cells[0:], name='adhesive')
    adhesivePart.SectionAssignment(region=adhesivePart.sets['adhesive'], sectionName='adhesive', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    emcPart.PartitionCellByPlanePointNormal(point=emcPart.vertices.findAt(coordinates=(l_die, l_die, h_adhesive + h_die)),
                                            normal=emcPart.edges.findAt(coordinates=(l_die, l_die, h_adhesive + h_die / 2)),
                                            cells=emcPart.cells.findAt(coordinates=(l_emc, l_emc, 0.)))
    emcPart.PartitionEdgeByParam(edges=emcPart.edges.findAt(((l_die, l_die, h_adhesive + h_die), (l_die, l_die, 0.))), parameter=h_adhesive / (h_die + h_adhesive))
    emcPart.PartitionCellByPlanePointNormal(point=emcPart.vertices.findAt(coordinates=(l_die, l_die, h_adhesive)),
                                            normal=emcPart.edges.findAt(coordinates=(l_die, l_die, h_adhesive + h_die / 2)),
                                            cells=emcPart.cells.findAt(coordinates=(l_emc, l_emc, 0.)))

    # Assembly
    subIns = a.Instance(name='sub', part=subPart, dependent=OFF)

    emcIns = a.Instance(name='emc', part=emcPart, dependent=OFF)
    a.translate(instanceList=('emc',), vector=(0.0, 0.0, h_sub))

    adhesiveIns = a.Instance(name='adhesive', part=adhesivePart, dependent=OFF)
    a.translate(instanceList=('adhesive',), vector=(0.0, 0.0, h_sub))

    dieIns = a.Instance(name='die', part=diePart, dependent=OFF)
    a.translate(instanceList=('die',), vector=(0.0, 0.0, h_sub + h_adhesive))

    a.InstanceFromBooleanMerge(name='merge', instances=(a.instances['sub'], a.instances['emc'], a.instances['adhesive'], a.instances['die'],),
                            keepIntersections=ON, originalInstances=DELETE, mergeNodes=BOUNDARY_ONLY, nodeMergingTolerance=1e-06, domain=BOTH)
    a.features.changeKey(fromName='merge-1', toName='merge')

    mergeIns = a.instances['merge']
    mergePart = m.parts['merge']
    mergePart.Set(cells=mergePart.cells[0:], name='all')
    mergePart.MaterialOrientation(region=mergePart.sets['all'], orientationType=GLOBAL, axis=AXIS_1, additionalRotationType=ROTATION_NONE, localCsys=None, fieldName='', stackDirection=STACK_3)
    mergePart.Set(edges=mergePart.edges.findAt(((l_die, 0, h_sub+h_adhesive),)), name='die_edge')
    mergePart.Set(edges=mergePart.edges.findAt(((l_adhesive, 0, h_sub),)), name='adhesive_edge')
    del m.parts['emc-1']
    del m.parts['die']
    del m.parts['adhesive']
    del m.parts['sub']

    # Mesh parts
    mergePart.seedPart(size=0.1, deviationFactor=0.1, minSizeFactor=0.1)
    mergePart.setMeshControls(regions=mergePart.cells[0:], algorithm=MEDIAL_AXIS)
    mergePart.generateMesh()
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    mergePart.setElementType(regions=mergePart.sets['all'], elemTypes=(elemType1, elemType2, elemType3))

    # Initial state definition
    m.Temperature(name='init_temp', createStepName='Initial', region=mergeIns.sets['all'],
                distributionType=UNIFORM, crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(175.0,))

    # Step definition
    m.StaticStep(name='Cooled', previous='Initial')
    m.Temperature(name='end_temp', createStepName='Cooled', region=mergeIns.sets['all'],
                distributionType=UNIFORM, crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(25.0,))

    # Boundary condition
    mergePart.Set(vertices=mergePart.vertices.findAt(((-l_sub, -l_sub, 0),)), name='xyz_corner')
    m.DisplacementBC(name='xyz_fixed', createStepName='Cooled', region=mergeIns.sets['xyz_corner'], u1=0.0, u2=0.0, u3=0.0, ur3=0., fixed=OFF, fieldName='', localCsys=None)
    mergePart.Set(vertices=mergePart.vertices.findAt(((l_sub, -l_sub, 0),)), name='xz_corner')
    m.DisplacementBC(name='xz_fixed', createStepName='Cooled', region=mergeIns.sets['xz_corner'], u2=0.0, u3=0.0, ur3=0., fixed=OFF, fieldName='', localCsys=None)
    mergePart.Set(vertices=mergePart.vertices.findAt(((-l_sub, l_sub, 0),)), name='yz_corner')
    m.DisplacementBC(name='yz_fixed', createStepName='Cooled', region=mergeIns.sets['yz_corner'], u1=0.0, u3=0.0, ur3=0., fixed=OFF, fieldName='', localCsys=None)
    mergePart.Set(vertices=mergePart.vertices.findAt(((l_sub, l_sub, 0),)), name='z_corner')
    m.DisplacementBC(name='z_fixed', createStepName='Cooled', region=mergeIns.sets['z_corner'], u3=0.0, ur3=0., fixed=OFF, fieldName='', localCsys=None)

    #NodeSets for postprocessing
    node1 = mergePart.nodes.getClosest((0, 0, 0))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='sub_center')
    node1 = mergePart.nodes.getClosest((0, 0, h_sub))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='sub_adhesive_center')
    node1 = mergePart.nodes.getClosest((0, 0, h_sub+h_adhesive))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='adhesive_die_center')
    node1 = mergePart.nodes.getClosest((0, 0, h_sub+h_adhesive+h_die))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='die_emc_center')
    node1 = mergePart.nodes.getClosest((0, 0, h_sub+h_emc))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_center')
    node1 = mergePart.nodes.getClosest((l_emc, l_emc, h_sub+h_emc))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_pp')
    node1 = mergePart.nodes.getClosest((l_sub, l_sub, 0))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='sub_pp')
    node1 = mergePart.nodes.getClosest((l_adhesive, l_adhesive, h_sub))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='sub_adhesive_pp')
    node1 = mergePart.nodes.getClosest((l_die, l_die, h_sub+h_adhesive))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='adhesive_die_pp')
    node1 = mergePart.nodes.getClosest((l_die, l_die, h_sub+h_adhesive+h_die))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='die_emc_pp')
    node1 = mergePart.nodes.getClosest((l_emc, l_emc, h_sub+h_adhesive+h_die))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_die_pp')
    node1 = mergePart.nodes.getClosest((l_emc, l_emc, h_sub+h_adhesive))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_adhesive_pp')
    node1 = mergePart.nodes.getClosest((l_emc, l_emc, h_sub))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_sub_pp')
    node1 = mergePart.nodes.getClosest((l_emc, 0, h_sub+h_emc))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_mid')
    node1 = mergePart.nodes.getClosest((l_adhesive, 0, h_sub))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='adhesive_sub_mid')
    node1 = mergePart.nodes.getClosest((l_die, 0, h_sub+h_adhesive))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='die_adhesive_mid')
    node1 = mergePart.nodes.getClosest((l_die, 0, h_sub+h_adhesive+h_die))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_die_mid')
    node1 = mergePart.nodes.getClosest((l_emc, 0, h_sub+h_adhesive))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_adhesive_mid')
    node1 = mergePart.nodes.getClosest((l_emc, 0, h_sub))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='emc_sub_mid')
    node1 = mergePart.nodes.getClosest((l_sub, 0, 0))
    mergePart.Set(nodes=mergePart.nodes[node1.label-1:node1.label], name='sub_mid')

    # Job setting
    job_name = model_name + '-emc_cte_' + str(emc_cte)
    emc_cte = emc_cte * cte_ratio
    m.materials['emc'].Expansion(type=ISOTROPIC, zero=25.0, temperatureDependency=ON, table=cte_temp(np.array([[.0, 25.0], [emc_cte, 112.2], [31e-6, 250.]])))
    mdb.save()
    mdb.Job(name=job_name, model=model_name, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
            memory=2, memoryUnits=GIGA_BYTES, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()

    # output checking
    o = session.openOdb(r'C:/SUSTech-Postdoc/SDIM-Projects/Warpage/Genetic_Algorithm/' + job_name + '.odb')
    oa = o.rootAssembly

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['SUB_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['SUB_PP']).values[0].data[2]
    sub_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['SUB_ADHESIVE_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['EMC_SUB_PP']).values[0].data[2]
    emc_sub_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['ADHESIVE_DIE_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['EMC_ADHESIVE_PP']).values[0].data[2]
    emc_adhesive_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['DIE_EMC_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['EMC_DIE_PP']).values[0].data[2]
    emc_die_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['EMC_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['EMC_PP']).values[0].data[2]
    emc_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['SUB_ADHESIVE_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['SUB_ADHESIVE_PP']).values[0].data[2]
    sub_adhesive_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['ADHESIVE_DIE_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['ADHESIVE_DIE_PP']).values[0].data[2]
    adhesive_die_warpage = v1 - v2

    v1 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['DIE_EMC_CENTER']).values[0].data[2]
    v2 = o.steps['Cooled'].frames[-1].fieldOutputs['U'].getSubset(region=o.rootAssembly.instances['MERGE'].nodeSets['DIE_EMC_PP']).values[0].data[2]
    die_emc_warpage = v1 - v2

    warpage = np.array([sub_warpage, emc_sub_warpage, emc_adhesive_warpage, emc_die_warpage, emc_warpage, sub_adhesive_warpage, adhesive_die_warpage, die_emc_warpage])
    np.savetxt('C:/SUSTech-Postdoc/SDIM-Projects/Warpage/Genetic_Algorithm/'+ job_name +'.csv', warpage, delimiter=',')
    o.close()
    return [h_die, h_adhesive, h_sub, h_emc, emc_cte, sub_warpage]

#%%
# Genetic algorithm
def ga_selection(pop_size = 10, population = [], variable_num = 4, fitness = [], mutation_ratio = 0.1):
    import numpy as np
    # Parents selection (max k)
    parent_size = pop_size//2
    parent = []
    for seq in range(parent_size):
        max_id = fitness.index(min(fitness))
        parent.append(population[max_id])
        del(fitness[max_id])
        del(population[max_id])

    # Crossover
    offspring =[]
    mating_seq = list(range(parent_size))
    random.shuffle(mating_seq)
    offspring = []
    for seq in range(parent_size):
        mating_rule = np.random.randint(2, size=variable_num)
        child = parent[seq] * mating_rule + parent[mating_seq[seq]] * abs(mating_rule - 1)
        offspring.append(child.tolist())
    random.shuffle(mating_seq)
    for seq in range(parent_size):
        mating_rule = np.random.randint(2, size=variable_num)
        child = parent[seq] * mating_rule + parent[mating_seq[seq]] * abs(mating_rule - 1)
        offspring.append(child.tolist())

    # Mutation
    mutation_num = int(len(offspring) * len(offspring[0]) * mutation_ratio)
    mutation_pos = []
    while len(mutation_pos) < mutation_num:
        temp = [random.randint(0, pop_size-1), random.randint(0, variable_num-1)]
        if temp not in mutation_pos:
            mutation_pos.append(temp)
    for pos in mutation_pos:
        mute_x = random.randint(0, variable_num-1)
        while mute_x == offspring[pos[0]][pos[1]]:
            mute_x = random.randint(0, variable_num-1)
        else:
            offspring[pos[0]][pos[1]] = mute_x
    return offspring

#%% 
pop_size = 10
init_pop = []
while len(init_pop) < pop_size:
    temp = [random.randint(0, j-1), random.randint(0, k-1), random.randint(0, l-1), random.randint(0, m-1)]
    if temp not in init_pop:
        init_pop.append(temp)
# run Abaqus simulation and collect initial result
sub_warpage_list = []
result_list = []
for item in init_pop:
    result = abaqus_model(item)
    result_list.append(result)
    sub_warpage_list.append(result[-1])
sub_warpage_abs = [abs(i) for i in sub_warpage_list]
np.savetxt('C:/SUSTech-Postdoc/SDIM-Projects/Warpage/Genetic_Algorithm/init_data_GA.csv', np.array(result_list), delimiter=',')

generation_num = 10
for generation in range(generation_num):
    offspring = ga_selection(pop_size = pop_size, population = init_pop, variable_num = variable_num, fitness = sub_warpage_abs, mutation_ratio = 0.5)
    print(offspring)
    result_list = []
    sub_warpage_list = []
    for item in offspring:
        result = abaqus_model(item)
        result_list.append(result)
        sub_warpage_list.append(result[-1]) 
    sub_warpage_abs = [abs(n) for n in sub_warpage_list]
    np.savetxt('C:/SUSTech-Postdoc/SDIM-Projects/Warpage/Genetic_Algorithm/Generation_' + str(generation + 1) + '.csv', np.array(result_list), delimiter=',')
    init_pop = offspring


# %%
