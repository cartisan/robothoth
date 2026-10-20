## Approaches
- **Dogfood products.** The term “dogfooding” is the name of the common practice of devs and employees using a product while they are building it, pre-release.
- **Test automation.** Run CI/CD on all environments including production.
- **Bug reports by users**. Build a relationship with users who get invested in the product and the reporting of bugs.
- **Alpha and beta testing.** Alpha and beta testing is about giving customers access to unfinished, less stable versions of a product. “Alpha” usually refers to a latest build that has had little to no QA testing. “Beta” versions have had some testing, but not as much as a full release.
- **Code reviews.** Spotting bugs is one of the purposes of code reviews, although they are not the best way to do so.
## Bug reporting
Features of useful bug reports:
- Useful metadata (e.g. version, device, system metrics)
- Relevant context (e.g. on mobile while connected to bluetooth speaker and poor connectivity, on a server in this region during lunch hour, on a debug build with these feature flags active, etc)
- Straightforward to reproduce, or have reproduction steps
- Reported by users who trust a reported bug will be fixed

Policies:
- Make it easy to create *quality* bugs reports
	- Don't force users to use tech jargon
	- Either collect relevant metadata yourself, or make it easy for users to find it
- Have an internal channel for bug reports
- Have a process that tracks bugs, and their current state


##
Associations: [[Software Engineering]], [[Giving MR Reviews]]
Tags: #🌱 
Sources: https://newsletter.pragmaticengineer.com/p/bug-management-that-works-part-1?
