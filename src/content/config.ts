import { defineCollection, z } from "astro:content";
import type { CollectionConfig } from "astro/content/config";

type PostsSchema = z.ZodObject<
	{
		title: z.ZodString;
		published: z.ZodDate;
		updated: z.ZodOptional<z.ZodDate>;
		draft: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
		pin: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
		description: z.ZodDefault<z.ZodOptional<z.ZodString>>;
		image: z.ZodDefault<z.ZodOptional<z.ZodString>>;
		tags: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodString>>>;
		category: z.ZodDefault<z.ZodNullable<z.ZodOptional<z.ZodString>>>;
		lang: z.ZodDefault<z.ZodOptional<z.ZodString>>;
		prevTitle: z.ZodDefault<z.ZodString>;
		prevSlug: z.ZodDefault<z.ZodString>;
		nextTitle: z.ZodDefault<z.ZodString>;
		nextSlug: z.ZodDefault<z.ZodString>;
	},
	"strip"
>;

const postsCollection = defineCollection({
	schema: z.object({
		title: z.string(),
		published: z.date(),
		updated: z.date().optional(),
		draft: z.boolean().optional().default(false),
		pin: z.boolean().optional().default(false),
		description: z.string().optional().default(""),
		image: z.string().optional().default(""),
		tags: z.array(z.string()).optional().default([]),
		category: z.string().optional().nullable().default(""),
		lang: z.string().optional().default(""),

		/* For internal use */
		prevTitle: z.string().default(""),
		prevSlug: z.string().default(""),
		nextTitle: z.string().default(""),
		nextSlug: z.string().default(""),
	}),
});
const specCollection = defineCollection({
	schema: z.object({}),
});
export const collections: {
	posts: CollectionConfig<PostsSchema>;
	spec: CollectionConfig<z.ZodObject<Record<never, never>, "strip">>;
} = {
	posts: postsCollection,
	spec: specCollection,
};
