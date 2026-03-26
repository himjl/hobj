
We note the following changes since this codebase was released alongisde the paper:


## 1/27/2025: 
* Benchmarks now report MSEn scores which subtract the sampling variance of the human target, meaning that a "correct model" will have an expected score of 0. Before, these benchmarks did not perform this subtraction, meaning a "correct model" would have an expected score equal to the human sampling variance.
* Statistical analyses other than the calculation of MSEn scores, their 95% confidence intervals, and generation of bootstrap resamples have been removed from this codebase.  
* Experiment 2: The lapse-rate corrected performance is now clipped between [0, 1].
* Experiment 2: The estimation of the variance of a lapse-rate corrected performance estimate from using bootstrapping to using the formula $\frac{1}{1-\hat p_{l}} \hat \sigma^2$, which treats the estimated lapse rate as a constant. Because a high number (~1000s) of Binomial observations are used to estimate $\hat p_{l}$, we consider this a decent approximation that is less computationally expensive.

## 3/25/2026 - cleaner data packaging



### Object renaming
I renamed the objects to follow a cleaner naming scheme. See below for the map from old names to new names:

<details>
<summary>High-variation object renames</summary>

```json
{
  "MutatorB2000_0": "MutatorObject000",
  "MutatorB2000_19": "MutatorObject001",
  "MutatorB2000_84": "MutatorObject002",
  "MutatorB2000_91": "MutatorObject003",
  "MutatorB2000_93": "MutatorObject004",
  "MutatorB2000_98": "MutatorObject005",
  "MutatorB2000_104": "MutatorObject006",
  "MutatorB2000_133": "MutatorObject007",
  "MutatorB2000_149": "MutatorObject008",
  "MutatorB2000_160": "MutatorObject009",
  "MutatorB2000_172": "MutatorObject010",
  "MutatorB2000_174": "MutatorObject011",
  "MutatorB2000_185": "MutatorObject012",
  "MutatorB2000_188": "MutatorObject013",
  "MutatorB2000_190": "MutatorObject014",
  "MutatorB2000_196": "MutatorObject015",
  "MutatorB2000_200": "MutatorObject016",
  "MutatorB2000_206": "MutatorObject017",
  "MutatorB2000_214": "MutatorObject018",
  "MutatorB2000_218": "MutatorObject019",
  "MutatorB2000_229": "MutatorObject020",
  "MutatorB2000_247": "MutatorObject021",
  "MutatorB2000_348": "MutatorObject022",
  "MutatorB2000_367": "MutatorObject023",
  "MutatorB2000_376": "MutatorObject024",
  "MutatorB2000_385": "MutatorObject025",
  "MutatorB2000_417": "MutatorObject026",
  "MutatorB2000_419": "MutatorObject027",
  "MutatorB2000_450": "MutatorObject028",
  "MutatorB2000_455": "MutatorObject029",
  "MutatorB2000_470": "MutatorObject030",
  "MutatorB2000_475": "MutatorObject031",
  "MutatorB2000_487": "MutatorObject032",
  "MutatorB2000_527": "MutatorObject033",
  "MutatorB2000_561": "MutatorObject034",
  "MutatorB2000_577": "MutatorObject035",
  "MutatorB2000_578": "MutatorObject036",
  "MutatorB2000_609": "MutatorObject037",
  "MutatorB2000_631": "MutatorObject038",
  "MutatorB2000_645": "MutatorObject039",
  "MutatorB2000_656": "MutatorObject040",
  "MutatorB2000_671": "MutatorObject041",
  "MutatorB2000_693": "MutatorObject042",
  "MutatorB2000_695": "MutatorObject043",
  "MutatorB2000_697": "MutatorObject044",
  "MutatorB2000_700": "MutatorObject045",
  "MutatorB2000_722": "MutatorObject046",
  "MutatorB2000_770": "MutatorObject047",
  "MutatorB2000_772": "MutatorObject048",
  "MutatorB2000_774": "MutatorObject049",
  "MutatorB2000_797": "MutatorObject050",
  "MutatorB2000_798": "MutatorObject051",
  "MutatorB2000_812": "MutatorObject052",
  "MutatorB2000_819": "MutatorObject053",
  "MutatorB2000_820": "MutatorObject054",
  "MutatorB2000_823": "MutatorObject055",
  "MutatorB2000_856": "MutatorObject056",
  "MutatorB2000_876": "MutatorObject057",
  "MutatorB2000_899": "MutatorObject058",
  "MutatorB2000_902": "MutatorObject059",
  "MutatorB2000_938": "MutatorObject060",
  "MutatorB2000_963": "MutatorObject061",
  "MutatorB2000_972": "MutatorObject062",
  "MutatorB2000_1012": "MutatorObject063",
  "MutatorB2000_1018": "MutatorObject064",
  "MutatorB2000_1024": "MutatorObject065",
  "MutatorB2000_1030": "MutatorObject066",
  "MutatorB2000_1034": "MutatorObject067",
  "MutatorB2000_4193": "MutatorObject068",
  "MutatorB2000_4235": "MutatorObject069",
  "MutatorB2000_4255": "MutatorObject070",
  "MutatorB2000_4259": "MutatorObject071",
  "MutatorB2000_4267": "MutatorObject072",
  "MutatorB2000_4287": "MutatorObject073",
  "MutatorB2000_4370": "MutatorObject074",
  "MutatorB2000_4448": "MutatorObject075",
  "MutatorB2000_4451": "MutatorObject076",
  "MutatorB2000_4454": "MutatorObject077",
  "MutatorB2000_4482": "MutatorObject078",
  "MutatorB2000_4489": "MutatorObject079",
  "MutatorB2000_4491": "MutatorObject080",
  "MutatorB2000_4521": "MutatorObject081",
  "MutatorB2000_4530": "MutatorObject082",
  "MutatorB2000_4567": "MutatorObject083",
  "MutatorB2000_4568": "MutatorObject084",
  "MutatorB2000_4569": "MutatorObject085",
  "MutatorB2000_4578": "MutatorObject086",
  "MutatorB2000_4614": "MutatorObject087",
  "MutatorB2000_4622": "MutatorObject088",
  "MutatorB2000_4630": "MutatorObject089",
  "MutatorB2000_4634": "MutatorObject090",
  "MutatorB2000_4635": "MutatorObject091",
  "MutatorB2000_4678": "MutatorObject092",
  "MutatorB2000_4710": "MutatorObject093",
  "MutatorB2000_4715": "MutatorObject094",
  "MutatorB2000_4718": "MutatorObject095",
  "MutatorB2000_4720": "MutatorObject096",
  "MutatorB2000_4723": "MutatorObject097",
  "MutatorB2000_4724": "MutatorObject098",
  "MutatorB2000_4732": "MutatorObject099",
  "MutatorB2000_4750": "MutatorObject100",
  "MutatorB2000_4752": "MutatorObject101",
  "MutatorB2000_4754": "MutatorObject102",
  "MutatorB2000_4772": "MutatorObject103",
  "MutatorB2000_4805": "MutatorObject104",
  "MutatorB2000_4821": "MutatorObject105",
  "MutatorB2000_4829": "MutatorObject106",
  "MutatorB2000_4832": "MutatorObject107",
  "MutatorB2000_4835": "MutatorObject108",
  "MutatorB2000_4845": "MutatorObject109",
  "MutatorB2000_4847": "MutatorObject110",
  "MutatorB2000_4872": "MutatorObject111",
  "MutatorB2000_4878": "MutatorObject112",
  "MutatorB2000_4908": "MutatorObject113",
  "MutatorB2000_4910": "MutatorObject114",
  "MutatorB2000_4911": "MutatorObject115",
  "MutatorB2000_4932": "MutatorObject116",
  "MutatorB2000_4939": "MutatorObject117",
  "MutatorB2000_4940": "MutatorObject118",
  "MutatorB2000_4945": "MutatorObject119",
  "MutatorB2000_4946": "MutatorObject120",
  "MutatorB2000_4950": "MutatorObject121",
  "MutatorB2000_4951": "MutatorObject122",
  "MutatorB2000_4953": "MutatorObject123",
  "MutatorB2000_4968": "MutatorObject124",
  "MutatorB2000_4974": "MutatorObject125",
  "MutatorB2000_4986": "MutatorObject126",
  "MutatorB2000_4998": "MutatorObject127"
}
```

</details>

<details>
<summary>One-shot object renames</summary>

```json
{
  "MutatorB2000_46": "MutatorOneshotObject00",
  "MutatorB2000_116": "MutatorOneshotObject01",
  "MutatorB2000_138": "MutatorOneshotObject02",
  "MutatorB2000_270": "MutatorOneshotObject03",
  "MutatorB2000_288": "MutatorOneshotObject04",
  "MutatorB2000_296": "MutatorOneshotObject05",
  "MutatorB2000_462": "MutatorOneshotObject06",
  "MutatorB2000_496": "MutatorOneshotObject07",
  "MutatorB2000_613": "MutatorOneshotObject08",
  "MutatorB2000_663": "MutatorOneshotObject09",
  "MutatorB2000_694": "MutatorOneshotObject10",
  "MutatorB2000_701": "MutatorOneshotObject11",
  "MutatorB2000_746": "MutatorOneshotObject12",
  "MutatorB2000_801": "MutatorOneshotObject13",
  "MutatorB2000_926": "MutatorOneshotObject14",
  "MutatorB2000_953": "MutatorOneshotObject15",
  "MutatorB2000_1164": "MutatorOneshotObject16",
  "MutatorB2000_1219": "MutatorOneshotObject17",
  "MutatorB2000_1229": "MutatorOneshotObject18",
  "MutatorB2000_1251": "MutatorOneshotObject19",
  "MutatorB2000_1258": "MutatorOneshotObject20",
  "MutatorB2000_1280": "MutatorOneshotObject21",
  "MutatorB2000_1363": "MutatorOneshotObject22",
  "MutatorB2000_1424": "MutatorOneshotObject23",
  "MutatorB2000_1767": "MutatorOneshotObject24",
  "MutatorB2000_1825": "MutatorOneshotObject25",
  "MutatorB2000_1865": "MutatorOneshotObject26",
  "MutatorB2000_2092": "MutatorOneshotObject27",
  "MutatorB2000_2106": "MutatorOneshotObject28",
  "MutatorB2000_2122": "MutatorOneshotObject29",
  "MutatorB2000_2130": "MutatorOneshotObject30",
  "MutatorB2000_2139": "MutatorOneshotObject31",
  "MutatorB2000_2198": "MutatorOneshotObject32",
  "MutatorB2000_2292": "MutatorOneshotObject33",
  "MutatorB2000_2304": "MutatorOneshotObject34",
  "MutatorB2000_2314": "MutatorOneshotObject35",
  "MutatorB2000_2344": "MutatorOneshotObject36",
  "MutatorB2000_2365": "MutatorOneshotObject37",
  "MutatorB2000_2444": "MutatorOneshotObject38",
  "MutatorB2000_2722": "MutatorOneshotObject39",
  "MutatorB2000_2757": "MutatorOneshotObject40",
  "MutatorB2000_2832": "MutatorOneshotObject41",
  "MutatorB2000_2909": "MutatorOneshotObject42",
  "MutatorB2000_3035": "MutatorOneshotObject43",
  "MutatorB2000_3043": "MutatorOneshotObject44",
  "MutatorB2000_3066": "MutatorOneshotObject45",
  "MutatorB2000_3077": "MutatorOneshotObject46",
  "MutatorB2000_3123": "MutatorOneshotObject47",
  "MutatorB2000_3278": "MutatorOneshotObject48",
  "MutatorB2000_3308": "MutatorOneshotObject49",
  "MutatorB2000_3496": "MutatorOneshotObject50",
  "MutatorB2000_3525": "MutatorOneshotObject51",
  "MutatorB2000_3527": "MutatorOneshotObject52",
  "MutatorB2000_3585": "MutatorOneshotObject53",
  "MutatorB2000_3601": "MutatorOneshotObject54",
  "MutatorB2000_3615": "MutatorOneshotObject55",
  "MutatorB2000_3636": "MutatorOneshotObject56",
  "MutatorB2000_3733": "MutatorOneshotObject57",
  "MutatorB2000_4049": "MutatorOneshotObject58",
  "MutatorB2000_4256": "MutatorOneshotObject59",
  "MutatorB2000_4305": "MutatorOneshotObject60",
  "MutatorB2000_4628": "MutatorOneshotObject61",
  "MutatorB2000_4703": "MutatorOneshotObject62",
  "MutatorB2000_4792": "MutatorOneshotObject63"
}
```

</details>


<details>
<summary>Warmup object renames</summary>

```json
{
  "Mutator19": "MutatorWarmupObject0",
  "Mutator20": "MutatorWarmupObject1",
  "Mutator21": "MutatorWarmupObject2",
  "Mutator22": "MutatorWarmupObject3",
  "Mutator25": "MutatorWarmupObject4",
  "Mutator26": "MutatorWarmupObject5",
  "Mutator29": "MutatorWarmupObject6",
  "Mutator30": "MutatorWarmupObject7"
}
```

</details>



### Image renaming

Following
