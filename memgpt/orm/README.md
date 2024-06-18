
### ORM Basic Design Patterns

Standard 3nf ORM pattern.
- first-class entities get their own module. These Modules with a single entity are named singularly ("user", "agent" etc) because they represent one model.
- Polymorphic models share a module, and these module names are pluralized ("Memories") because there are multiple.
- mixins, helpers, errors etc are collections so the module names are pluralized.
- Imports are always lazy whenever possible to guard against circular deps.


## Mixin magic
relationship mixins expect standard naming (ie `_organization_id` on the child side of 1:M to an organization).

The relationship is declared explicitly in the model, as the lazy joining rules, back populates etc will be bespoke per class.

If you need to reference the same entity more than once, you'll need to skip the mixin and do it by hand.