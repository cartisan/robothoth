## TL;DR

_Whole section to be filled out in the end._

**Incident summary**

Write a summary of the incident in a few sentences. Include what happened, why, the severity of the incident and how long the impact lasted.

**Impact**

Describe how the incident impacted internal and external users.

**Root cause**

Note the final root cause of the incident, the thing identified that needs to change in order to prevent this class of incident from happening again.

## Details

### Leadup

Describe the sequence of events that led to the incident, for example, previous changes that introduced bugs that had not yet been detected.

### Fault

Describe how the change that was implemented didn't work as expected. If available, attach screenshots of relevant data visualizations that illustrate the fault.

### Detection

When did the team detect the incident? How did they know it was happening? How could we improve time-to-detection? Consider: How would we have cut that time by half?

### Response

Who responded to the incident? When did they respond, and what did they do? Note any delays or obstacles to responding.

### Recovery

Describe how the service was restored and the incident was deemed over. Detail how the service was successfully restored and what steps were needed to take to recovery.

Depending on the scenario, consider these questions: How could we improve time to mitigation?

### Timeline

Detail the incident timeline. Include any notable lead-up events, any starts of activity, the first known impact, and escalations. Note any decisions or changes made, and when the incident ended, along with any post-impact events of note.

XX:XX UTC - INCIDENT ACTIVITY; ACTION TAKEN

XX:XX UTC - INCIDENT ACTIVITY; ACTION TAKEN

XX:XX UTC - INCIDENT ACTIVITY; ACTION TAKEN

## Root cause identification

### The Five Whys

The Five Whys is a [root cause identification technique](https://www.atlassian.com/team-playbook/plays/5-whys). Here’s how you can use it:

- Begin with a description of the impact and ask why it occurred.
- Note the impact that it had.
- Ask why this happened, and why it had the resulting impact.
- Then, continue asking “why” until you arrive at a root cause.

List the "whys" in your postmortem documentation.

### Recurrence

Now that you know the root cause, can you look back and see any other incidents that could have the same root cause? If yes, note what mitigation was attempted in those incidents and ask why this incident occurred again.

## Lessons learned

### Evaluation

Discuss what went well in the incident response, what could have been improved, and where there are opportunities for improvement.

### Backlog check

Review your engineering backlog to find out if there was any unplanned work there that could have prevented this incident, or at least reduced its impact? A clear-eyed assessment of the backlog can shed light on past decisions around priority and risk.

### Corrective actions

Describe corrective actions that need to be taken to prevent this class of incident in the future. Note who is responsible, when they have to complete the work and where that work is being tracked.
