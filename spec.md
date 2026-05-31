AI Capstone Spring 2026

 Course Project Specification

Deadline: 2026/06/07 23:59

Overview

The  AI  Capstone  Course  Project  is  designed  to  provide  students  with  hands-on
experience  across  the  full  lifecycle  of  Physical  AI—from real-world data collection,
to policy learning, and ultimately to evaluation.

To support diverse backgrounds, learning goals, and levels of ambition, the course is
structured  into  three  progressive  tiers: Entry, Advanced, and Independent Study.
Each  level  features  distinct  learning  activities  and  assessment  criteria,  enabling
students to engage meaningfully while pursuing increasing levels of technical depth,
creativity, and research-oriented exploration.

This  tiered  design  not  only  builds  strong  foundational  competence  but  also
establishes  a  clear  pathway  from  structured learning to open-ended innovation and
real-world  deployment.  By  tightly  integrating  simulation,  learning,  and  physical
system  validation,  the  course  project  offers  a  unified  and  distinctive  framework  for
training the next generation of Physical AI talents.

Course Project Hierarchy

1

This  project  consists  of both group and individual components. The Entry Level
is  a  mandatory  group  project  that  accounts  for  80%  of  the  final  grade.  The
Advanced Level is an optional extension of the group work, providing an additional
20%  on  top  of  the  Entry  score.  The  Independent  Study  Level  is  an  optional
individual component that offers up to 30% bonus. In other words, completing both
the Entry and Advanced group components allows you to achieve up to 100%
of  the  final  project  grade. If you also complete the individual component, you
can earn up to 130% in total. In addition, there is a leaderboard mechanism in the
Entry Level: the group with the highest performance in each task will receive an extra
5% bonus score.

Group Project ------------------------------------------------------

Entry Level (80%)
(Please note that participation at this level is mandatory, and all students must join a group.)

At  the  Entry  level,  students  will  experience  the  full  lifecycle  of  Physical  AI  within  a
predefined tasks, training, and evaluation framework. Given the practical constraints
in  accessing  robotic  systems  capable  of  real-world interaction through sensors and
actuators,  we  develop  a  Robot  Learning  framework
that  emulates  such
interactions within a physics-based simulation environment. This framework enables
students  to  experience  both  perception  and  action  loops  through  embodied
interaction between a robotic agent and a simulated environment.

The framework provides:

●  a base simulation environment built on NVIDIA Isaac Sim [5]
●  a suite of baseline manipulation tasks (e.g., object picking and placing)
●  reference policies integrated from the LeRobot ecosystem [4]
●  and standardized evaluation metrics (e.g., task success rate)

This  unified  setup  lowers  the  barrier  to  entry  while  preserving  the  essential
characteristics  of  Physical  AI  systems, allowing students to focus on understanding
core principles such as perception-action loop, policy learning, and policy evaluation.

Each  group  should  select  and  complete  one  of  three  provided  tasks  (see  the
following section for details), on a first-come, first-served basis via a shared sign-up
sheet.

Within  this  level,  each  group  is  allowed  to  explore  and  modify  the  following  two
aspects: training data and policy model. Please note that any changes beyond these
two aspects are not allowed:

●  Training data: including the data itself (i.e., what constitutes good or suitable
data),  dataset  size  (i.e.,  how  much  data  is  needed),  and  data  collection
methods.

○  At the Entry level, we provide two data collection methods: UMI [1] with
motion  planning  and  keyboard  control,  as  detailed  in  the  following

2

section.  Students  are  expected  to  explore  how  much  data  is  needed
under limited time and computational resources. To give you hands-on
experience with real-world data collection, a time slot will be scheduled
for each group to collect data using UMI in ED305.

●  Policy  model:

including  model

corresponding
hyperparameters. You are encouraged to explore different model designs and
tuning strategies. The LeRobot ecosystem [4] used in this project can be used
as a reference.

architecture

and

Students  will  participate  in  a  two-stage  evaluation  process  involving  Public  and
Private  Leaderboards.  During  the  active  development  phase,  a  public  validation
configuration  will  be  provided  to track real-time performance. The final rankings will
to  ensure  model
be  determined  by  a  separate  private
the  release  of  both
integrity.  Further  details  regarding
generalization  and
leaderboards will be shared in subsequent announcements.

test  configuration

Grading Policy:

Component

Technical Report

Presentation Video

Weight (in final grade)

40%

40%

The top-performing group in each task

+5% (on top of Entry level)

Advanced Level (+20%, Optional)
(Participation in this level is optional and left to each group’s discretion.)

The  Advanced  level  encourages  deeper  exploration  and  innovation  by  allowing
students  to  define  new  tasks  beyond  the  Entry  level,  including  task  definition,
environment settings, and evaluation metrics. Supporting resources (e.g., 3D assets
and  reference  implementations)  are  provided  in  X-Humanoid,  Synthesis,  and  USD
Assets  Working  Group  [6-8]  to  facilitate  creative  and  technically  rigorous  project
extensions. Please note that all work at this level is conducted in simulation.

Students  MUST  complete the Entry level before progressing to the Advanced level.
Both  levels  share  a  unified development framework, enabling a seamless transition
from structured learning to open-ended research and system design.

This level emphasizes:

●  Originality and innovation
●  Problem formulation
●  End-to-end system design and evaluation

Students are encouraged to explicitly address sim-to-real considerations and discuss
potential  real-world  deployment  challenges.  Projects  at  this  level  are  expected  to
approach the scope, rigor, and quality of a workshop-level research contribution.

3

Grading Policy:

Component

Technical Report

Weight (in final grade)

+10% (on top of Entry level)

Presentation Video

+10% (on top of Entry level)

Reproducibility Requirements:

At  this  advanced  level,  students  are  responsible  for  defining  their  own  tasks.  To
include  a  comprehensive
ensure  external  validation,  submissions  must
implementation  guideline.  The  provided  document  should  outline  step-by-step
instructions that allow our team to reproduce the claimed results.

●  Code submission with clear documentation
●  Fixed evaluation protocol (e.g., random seeds, dataset splits)
●  Reproducibility checklist to ensure consistent benchmarking

Individual Project -------------------------------------------------

Independent Study Level (+ 30%, Optional)

The Independent Study level offers a unique opportunity for students to participate in
ongoing  research-oriented  projects  in  collaboration  with  members  of  the  HCIS
Lab.

Students will have the opportunity to engage in frontier research, collaborate closely
with  researchers,  and  gain  hands-on  experience  with  advanced  robotic  systems,
experimental  platforms,  and  evaluation.  This  includes  exposure  to  real-world
challenges such as perception uncertainty, system integration, safety considerations,
human–robot interaction, and ontology-based data verification.

This  level  is  designed  for  highly  motivated  students  seeking  deeper  research
engagement  and  is  intended  to  bridge  coursework  with cutting-edge academic and
real-world applications.

Research Projects

Project Name

Mentor

Knowledge Representation and Formal Ontology for PhyAI  張俊彦

Agentic Mobile Manipulation

Adaptive Policy Learning for Meal Preparation

ToddlerBot

林翔恩

陳晉祿

陳奕廷

4

Project Name

Robotic Food Acquisition for Assistive Feeding

Real-World Bimanual Manipulation

Mentor

戴嬿玲

林谷翰、黃毓翔、鄔仁迪、
周士傑

Vision Language Model-enabled UAV Autonomy

曾子昕

Safe and Convenient Object Handover for In-Home
Assistive Manipulation

張家睿、蔡茗鈞

Grading Policy:

Component

Technical Report

Mentor’s Endorsement

Weight (in final grade)

+20%

+10%

●  Technical  Report:  A  comprehensive  report  documenting  the  project  scope,
methodology,  experimental  results,  and  critical  analysis.  The  report  should
clearly  articulate problem formulation, system design, implementation details,
and reflections on real-world deployment challenges.

●  Mentor’s Endorsement:

○  Process  Engagement:  Evaluation  of  the  student’s  engagement  and
contribution
throughout  the  research  process,  including  initiative,
collaboration  with  mentors,  consistency  of  progress,  and  ability  to
incorporate feedback.

○  Project-Specific  Components:  Additional  evaluation  components
may be incorporated at the discretion of the project mentor, depending
on
the  domain  and  project  scope.  These  may  include  system
demonstrations,  experimental  validations,  prototype  development,  or
other research-oriented deliverables.

A Robot Learning Framework

In  this  section,  we  present  a  detailed  description  of  the  robot  learning  framework
used  in  this  course  project  for  the  Entry  level.  An  overview  of  the  framework  is
illustrated below.

Overview

5

None

Real-World Data Collection with UMI

    ↓
   Trajectory Extraction and Reconstruction in Simulation

    ↓
Robot Motion Generation and Data Creation

    ↓
   Policy Training

    ↓
    Inference and Evaluation

Step 1. Real-World Data Collection with UMI

To  provide  hands-on  experience  with  real-world  data  collection,  we  employ  the
Universal  Manipulation  Interface  (UMI)  [1]  as  a  hand-held  device  to  collect  human
demonstration  data  in  the  ED305  classroom.  This  process  enables  students  to
capture natural, task-relevant behaviors that serve as the foundation for subsequent
policy learning.

Each group is assigned one of the following pick-and-place tasks:

●  Cup Stacking (Kitchen Scenario):

Randomly  place  a  blue  cup  and  a  pink  cup  upside  down  on the countertop.
The task is to pick up the blue cup and stack it on top of the pink cup.

●  Cutlery Arrangement (Dining Room Scenario):

Randomly  place  a  fork  and  a  knife  on  the  table.  The  task  is  to  position  the
knife on the right side of the plate and the fork on the left side.

●  Toy Block Collection (Living Room Scenario):

Randomly  scatter  three  toy  blocks  on  the  tabletop.  The  task  is  to  collect  all
scattered blocks and place them into a designated basket.

For more detailed information, please refer to the UMI Data Collection Guideline.

6

Figure 1. Illustrations of the predefined tasks in the ED305 classroom

Step 2. Reconstruction in Simulation

In this step, students reconstruct real-world demonstrations within NVIDIA Isaac Sim
[5]  through  a  provided  systematic  pipeline.  Due  to  practical  constraints,  we  do  not
provide  real  robots  at  this  level.  As  an  alternative,  we  employ  a  simulation
environment
testing  and  development,  while  improving
accessibility, scalability, and experimental reproducibility.

to  support  extensive

To  bridge
the  gap  between  real-world  data  collection  and  simulation-based
experimentation,  the  collected  demonstrations  are  reconstructed  in  the  simulation
environment. In this process, only object poses from the real-world data are used
to  set  up  the  simulation scene, while the recorded human trajectories are not
directly transferred. This design choice is due to several practical limitations:

●  Noise and inaccuracies in trajectory extraction from demonstrations
●  Errors in spatial configuration estimation
●  The sim-to-real gap in dynamics and control
●  Complex physical interactions and collision handling in simulation
●  Variability in embodiment (e.g., differences between human motion and robot

kinematics)

Instead  of  replaying  raw  human  trajectories, a motion planning algorithm is used to
generate  feasible  robot  actions  in  simulation,  as  discussed  in  the  next  section.
Transferring
remains
challenging.  Students  can  further  explore  improvements  to  this  pipeline,  and
promising solutions will be incorporated into the course repository.

real-world  human  motion

into  simulation

trajectories

Step 3. Robot Motion Generation and Data Creation

Following  the  reconstruction  process,  robot  motions  are  generated  using  a  finite
state  machine–based  planner  [9,  10].  Given  the  reconstructed  object  states  and

7

task  configurations,  the  planner  produces  feasible  action  sequences  that  satisfy
kinematic and environmental constraints within the simulation.

In addition to the planner-based pipeline, keyboard teleoperation is available as an
alternative  method for collecting demonstration data in simulation. It can be used to
supplement  real-world  data  when  it  is  insufficient,  or  to  support  extended  data
collection  in  the  Advanced  level.  Students  may  also  implement  their  own  data
collection methods as an extension.

To  ensure  data  quality,  explicit task success criteria (e.g., correct object placement,
completion  within  constraints)  are  defined  and  applied  to  all  collected  trajectories.
Only  successful  executions  are  recorded  and  stored  as  valid  training  data.  The
resulting  dataset  is  organized  in  the  format  defined  by  the  open-sourced  LeRobot
ecosystem [4], enabling efficient downstream policy training.

Figure 2. Illustrations of the predefined tasks in the simulation environment, including
Cup Stacking (top), Toy Block Collection (middle), and Cutlery Arrangement
(bottom).

8

Extension for Advanced Level

In the Advanced level, students are expected to move beyond the predefined tasks,
objects, and environments introduced in the Entry level. Specifically, students will:

●  Design new tasks, environments, and object configurations
●  Develop customized data collection strategies aligned with their task

definitions

●  Generate or curate assets (e.g., using USD-based scene descriptions) to

support their experimental setup

While  existing  resources  [6-8]  could  be  used  as  a  starting  point,  students  are
encouraged  to  extend  beyond  them  and  construct  original,  well-motivated  problem
settings.  This  stage  emphasizes  end-to-end  system  thinking,  requiring  students  to
jointly consider task design, data generation, and policy learning.

Step 4. Policy Training

Using  the  dataset  generated  in  the  previous  step,  we  train  policies  to  map
observations  to  robot  actions.  We  leverage  a  suite  of  robot  learning  algorithms
provided  by  the  LeRobot  ecosystem  [4].  As  a  starting  point,  we  recommend
Diffusion  Policy  [3],  a  state-of-the-art  imitation  learning  approach  for  visuomotor
control. Given visual observations and robot states, the policy learns to predict action
sequences (i.e., robot trajectories).

Students are encouraged to:

●  Explore alternative policy architectures within or beyond the provided

framework.

●  Tune training strategies (e.g., data augmentation, normalization, horizon
length, etc.)Analyze the relationship between dataset quality and policy
performance.

This step emphasizes learning-based control, bridging perception and action through
a scalable policy learning framework.

Step 5. Inference and Evaluation

After training, policies are evaluated under varying object positions to assess their
robustness and generalization capabilities. The evaluation is conducted under two
modes: leaderboard and local evaluation.

●  Leaderboard: We provide public and private leaderboards that only allow N

submissions per day. More details will be announced later.
●  Local evaluation: Rollout the trained policy on your side.

9

Performance on the private leaderboard will be ranked separately for each task
based on predefined metrics. The top-performing group in each task will receive a
bonus 5 points toward the final course project score. Further details on evaluation
criteria, ranking methodology, and bonus allocation will be provided in a separate
document. The rules for the leaderboard may be adjusted on a rolling basis.

Recommendations

To help you get started effectively, consider the following key principles:

1. Data Quality is Critical

Model performance is fundamentally driven by the quality and relevance of the data
you  collect.  While  collecting  additional  real-world  data  outside  your  assigned  time
slot  is  not  permitted,  you  are  encouraged  to  improve  and  extend  the  data
generation pipeline to increase both the quantity and diversity of training data. Any
modifications must be clearly documented in both your report and presentation.

Possible directions include (but are not limited to):

●  Revisiting Real-World UMI Data Processing: Trajectory extraction from

hand-held devices such as UMI is inherently noisy and may not always yield
valid demonstrations. You are encouraged to:

○  Analyze the data processing pipeline (Visualize trajectories, inspect

failure cases, and compare successful vs. failed samples)
Identify sources of failure (e.g., tracking errors, pose estimation noise)

○
○  Propose improvements to enhance robustness

Improving this stage is particularly important, as the simulation pipeline is
highly dependent on real-world data quality.

●  Enhancing Simulation Data Generation: You may modify the existing
simulation pipeline to improve diversity and robustness, for example:

○
○

Introduce multiple grasp poses or randomized motion strategies
Inject controlled randomness in robot motion to improve generalization

●  Designing Custom Data Collection Pipelines: Implement own data
collection interfaces or pipelines to generate additional training data.

2. Hyperparameter Tuning Matters

Model  performance  is  highly  sensitive  to  hyperparameter  choices.  While  you  may
begin  with  default  training  settings,  we  strongly  recommend  consulting  the  original
Diffusion  Policy  [3]  paper  for  deeper  insights  into  effective  configurations.  In
particular,  pay  close  attention  to  hyperparameters  related  to  task  horizon  length,

10

temporal  consistency,  multi-step  behaviors,  and  action  representation.
Systematic  experimentation  with  different  parameter  combinations  is  strongly
encouraged, as it can significantly impact policy performance.

Platform and Resources

We  have  prepared  the  following  resources  to  support  you  in  completing  the  Entry
level:

●  Monorepo for AI Capstone (link): The main codebase for the final project.

●  Universal  Manipulation  Interface  (UMI)  (link):  A  data  collection and policy
learning  framework  used  in  this  project,  enabling  direct  skill  transfer  from
in-the-wild human demonstrations to deployable robot policies.

○  UMI  Data  Collection  (guide):  Instructions  for  using  UMI  to  collect

real-world data in ED305.

●  Final Project QA Page (link): Please feel free to ask questions.

●  Glows.AI  (guide):  This  platform  provides  GPU  resources.  We  will  assign
credits  to  the  team  leader  after  your  team  has  finished  collecting  real-world
data with UMI.

Submission

The  deadline  for  submission  of  the  project  is  06/07  (Sun.)  at  23:59.  Each  group
needs to submit the following to the assigned Google Drive folder:

●  Presentation  Video:  Video  named  Team{TEAM_ID}_presentation.mp4,  in
which  you  briefly  explain  your  project  work.  The  video  should  be  within  15
minutes.

●  Project  Report:  Report named Team{TEAM_ID}_project_entry_report.pdf.
Groups  that  have  completed  the  advanced  requirements  must  also  submit
Team{TEAM_ID}_project_advanced_report.pdf.

Both  reports  should  provide  a  detailed  description  of  your  implementation,
design  choices,  and  insights developed throughout the project. Please follow
the guidelines outlined in the “Report Format” section below.

●  Checkpoint Folder: A directory containing all trained model checkpoints.
Refer to docs/lerobot-model-format.md for required structure and
contents.

●  Configurations Folder: A directory containing all necessary environment and
task configuration files. See docs/standalone_env_config_export.md
for export requirements. This is required for Advanced level only.

11

●  Custom CAD Models: Submission of all 3D models used in the project,
which must be provided in the .usd file format. This is required for the
Advanced level only, if custom assets are used.

●  README.txt: A mandatory file clearly specifying the complete execution
guidelines for your code. For Advanced Level submissions, this file must
include a comprehensive implementation guide that enables the evaluation
team to independently reproduce all claimed results, explicitly stating if no
additional files or custom configurations are needed to ensure benchmarking
integrity.

●  Supplementary Files: Any additional supporting files necessary for

reproducibility or complete documentation of the project.

Note:

●  All  submission  files  must  be  uploaded  to  the  appointed  Google  Drive
folder.  A  form  has  been provided, which includes the Google Drive link
for each group. Please fill in the email address of one designated group
member  (team  leader)  for  each  group  in  the  form  by  May  11.  We  will
grant Google Drive access permission to that student on May 12 so they
can upload the files.

●  Once  the  submission  deadline  has  passed,  no  further  modifications  are
allowed.  If  any  file  is  found  to  have  a modification timestamp later than
the  deadline,  it  will  be  treated  as  a  late  submission  and  penalized
accordingly.
If any file within the submitted link cannot be opened or downloaded properly,
the part of submission will not be graded.

●

●  For this final project, we do not accept any late submissions.
●  Any formatting errors will receive a 10-point deduction.

Presentation Video Format

There  is  no  strict format that you must follow. You are encouraged to organize your
presentation in a clear manner that best communicates your work. All presentations
must be delivered in English. If you have completed the advanced level, include
it  in  the same video as the entry-level presentation. The presentation should be
within 15 minutes, and must address the following key aspects:

●  Task Description: Clearly describe the assigned task, problem setting, and

objectives.

●  Pipeline and Method Modifications: Explain any modifications or

improvements you made to the provided pipeline, models, or codebase.
●  Key Learnings and Insights: Reflect on what you learned from the project,
including challenges encountered, solutions explored, and key takeaways.

12

A well-structured presentation should emphasize clarity, technical depth, and critical
reflection, rather than simply reporting results.

Report Format

All groups must follow the required report structure:

●  Entry-level  work:  Complete  all  sections  without  the  “(Advanced)”  label.

Sections marked “(Advanced)” must not be included.

●  Advanced  work:  Only  groups  that  complete  the  Advanced  requirements

should include and complete the sections marked “(Advanced)”.

Group Report Template:

1. Task Description

Introduce the task you are working on.

●
●  (Advanced) Please describe the motivation of the proposed new task.

2. Data Collection

●  (Entry-Only) Real-World Data Collection with UMI

○  Explain how you collected the data using the UMI device.
○  Describe how the collected demonstrations were converted into a

training dataset.

○  Provide visualizations of the collected data.

●  Simulation Data Collection in Isaac Sim

○  Describe dataset size and the percentage of each collection methods
(UMI with finite state machine–based planner / keyboard teleoperation
/self-implemented method)

○  (Optional) If you have self-implemented data collection methods,

provide your justifications and implementation details.

○  Provide visualizations of the collected data with the provided tool.
○  (Advanced) Describe your changes and design compared to the Entry

level.

3. Policy Training

●  Model Architecture

○  Describe the model architecture used in your project.
○  (Optional) If you modified the original architecture, explain the changes

and your motivations and observations.

●  Training Procedure

○  Describe the overall training procedure.

13

○  List and explain the key hyperparameters.
○  (Optional) If you adopted a different training procedure, describe your

approach and the reasons for your design choices.

4. Experimental Results

●  Data Quality Analysis

○  Explore  and  identify  methods  to  analyze  your  collected  data,  and
determine whether it is of sufficient quality and suitable for use in robot
learning.

●  Quantitative and Qualitative Result of Policy Models

○  Figure  of  training  loss  curves  to  illustrate the learning process of your
models.  Describe  and  analyze  your  results,  including  performance
trends and convergence behavior.

○  Provide  the  calculated  success  rate over N rollout times on your side,
refer to “#Rollout section” of the readme in the “aicapstone” repository.

○  Provide screenshots of the rollout procedure of your trained policy.
○  Any you want to share.

(Advanced)  Students  are  expected  to  provide  a  detailed  description  of  their
performance evaluation process. Please include:

●  Evaluation Procedure: Clearly describe the evaluation criteria and

environment configurations, and explain the rationale behind their design.
●  Results: Present the outcomes of your evaluation using tables, figures, or

numerical results if applicable.

5. Discussion

Share your insights, challenges encountered during the project, and lessons learned.
You may also discuss potential improvements or future directions.

6. Work Distribution

●  Each team member’s assigned tasks and responsibilities.
●  Each team member’s percentage of work contribution.

Individual Report:

●  Please discuss and confirm the report format with your mentor.
●  Students who complete an individual project must submit a report to the E3

system under “Independent Study Report.”

References

14

[1]  Chi  et  al.,  Universal  manipulation  interface:  In-the-wild  robot  teaching  without
in-the-wild robots. RSS 2024. Project page: https://umi-gripper.github.io/
[2] Isaac Sim Document: ttps://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
[3]  Chi,  et  al.  Diffusion  policy:  Visuomotor  policy  learning  via  action  diffusion.  The
International  Journal  of  Robotics  Research,  44.10-11  (2025):  1684-1704.  Project
Page: https://diffusion-policy.cs.columbia.edu/
[4] LeRobot: https://huggingface.co/lerobot
[5] Isaac Lab: https://developer.nvidia.com/isaac/lab
[6] X-Humanoid/ArtVIP: Hugging Face

[7] Synthesis: https://synthesis.extwin.com/#/home

[8] USD Assets Working Group: https://github.com/usd-wg/assets/tree/main

[9] Finite-state machine: Finite-state machine Wiki

[10] Pick and Place implementation: IsaacSim’s Official Implementation

[11] genrobot2025/10Kh-RealOmin-OpenData: Hugging Face

15


