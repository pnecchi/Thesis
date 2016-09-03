#include <thesis/Task.h>

Task::Task(Environment const &environment_)
    : environmentPtr(environment_.clone())
{
    /* Nothing to do */
}

Task::Task(Task const &task_)
    : environmentPtr(task_.environmentPtr->clone())
{
    /* Nothing to do */
}

