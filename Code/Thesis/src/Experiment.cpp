#include "thesis/Experiment.h"

Experiment::Experiment(Task const &task_,
                       Agent const &agent_)
    : taskPtr(task_.clone()),
      agentPtr(agent_.clone())
{
    /* Nothing to do */
}

Experiment::Experiment(Experiment const &experiment_)
    : taskPtr(experiment_.taskPtr->clone()),
      agentPtr(experiment_.agentPtr->clone())
{
    /* Nothing to do */
}

