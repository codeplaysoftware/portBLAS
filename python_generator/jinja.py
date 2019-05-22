#!/usr/bin/env python3
import yaml
import jinja2
from argparse import ArgumentParser

def render(environment, model, template):
    @jinja2.contextfilter
    def jinja_eval(context, value, **kwargs):
        if kwargs:
            kwargs.update(context)
            ctx = kwargs
        else:
            ctx = context
        return jinja2.Template(value).render(ctx)

    env = jinja2.Environment()
    env.filters['eval'] = jinja_eval
    template = env.from_string(template_string)
    return template.render(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model",
                        help="YAML file containing the data model", metavar="model.yaml")
    parser.add_argument("-t", "--template", dest="template",
                        help="Template to fill out using the model", metavar="template.cpp.in")
    parser.add_argument("-o", "--output", dest="output",
                        help="", metavar="generated.cpp")

    args = parser.parse_args()

    with open(args.model) as model_file:
        model_string = yaml.safe_load(model_file)

    with open(args.template) as template_file:
        template_string = template_file.read()

    environment = jinja2.Environment()
    rendered = render(environment, model_string, template_string)

    with open(args.output, "w") as output_file:
        output_file.write(rendered)
